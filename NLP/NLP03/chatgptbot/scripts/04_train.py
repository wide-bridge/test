"""
Step 5-6: GPT(Decoder-only) 모델 학습
- <start>, <end>, <sep> 토큰 사용
- Q <sep> A 를 하나의 시퀀스로 결합 (next token prediction)
- RTX 4090 최적화 배치 사이즈 설정
- GPT 모델 정의 및 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.experiment_tracker import ExperimentTracker

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

class GPTDataset(Dataset):
    """GPT용 데이터셋 클래스 (next token prediction)"""

    def __init__(self, df, vocab, max_length=config.MAX_SEQ_LENGTH):
        self.questions = df['question'].values
        self.answers = df['answer'].values
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        q_tokens = question.split()
        a_tokens = answer.split()

        q_indices = self._tokens_to_indices(q_tokens)
        a_indices = self._tokens_to_indices(a_tokens)

        # <start> Q <sep> A <end> 형태로 결합
        start_id = self.vocab.get('<start>', 1)
        sep_id = self.vocab.get('<sep>', 1)
        end_id = self.vocab.get('<end>', 1)

        sequence = [start_id] + q_indices + [sep_id] + a_indices + [end_id]
        sequence = self._pad_or_trim(sequence, self.max_length + 1)  # +1: input/target shift용

        # input: sequence[:-1], target: sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'q_text': question,
            'a_text': answer
        }

    def _tokens_to_indices(self, tokens):
        """토큰을 인덱스로 변환"""
        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(self.vocab.get('<unk>', 1))
        return indices

    def _pad_or_trim(self, sequence, max_length):
        """시퀀스 패딩 또는 자르기"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = [self.vocab.get('<pad>', 0)] * (max_length - len(sequence))
            return sequence + padding

class GPTModel(nn.Module):
    """GPT (Decoder-only Transformer) 모델"""

    def __init__(self, vocab_size, d_model=config.TRANSFORMER_D_MODEL,
                 nhead=config.TRANSFORMER_NHEAD, num_layers=config.TRANSFORMER_NUM_LAYERS,
                 dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD, dropout=config.DROPOUT_RATE,
                 max_seq_length=config.MAX_SEQ_LENGTH):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, causal_mask=None):
        """
        Forward pass

        Args:
            x (Tensor): 입력 (batch_size, seq_len)
            padding_mask (Tensor): 패딩 마스크 (batch_size, seq_len)
            causal_mask (Tensor): causal mask (seq_len, seq_len)

        Returns:
            Tensor: 출력 (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # 토큰 임베딩 + 위치 임베딩
        x_emb = self.token_embedding(x) * np.sqrt(self.d_model)
        x_emb = x_emb + self.position_embedding(positions)
        x_emb = self.dropout(x_emb)

        # Decoder-only Transformer
        output = self.transformer(
            x_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        logits = self.fc_out(output)
        return logits

class Tokenizer:
    """토크나이저 클래스"""

    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {v: k for k, v in vocab.items()}

    def save(self, path):
        """토크나이저 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
        print(f"Tokenizer saved to {path}")

def create_vocab_from_corpus(corpus_path):
    """코퍼스에서 어휘와 메타데이터 반환"""
    with open(corpus_path, 'rb') as f:
        corpus_data = pickle.load(f)
    word2idx = corpus_data['word2idx']
    tokenizer_name = corpus_data.get('tokenizer_name', 'unknown')
    return word2idx, tokenizer_name

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """한 에포크 학습"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        # 패딩 마스크 (input 기준)
        padding_mask = (input_ids == 0)

        # Causal mask
        seq_len = input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Forward pass
        output = model(input_ids, padding_mask=padding_mask, causal_mask=causal_mask)

        # loss 계산 (패딩 토큰 제외)
        loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # OneCycleLR은 배치마다 step() 호출해야 함
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            padding_mask = (input_ids == 0)

            seq_len = input_ids.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            output = model(input_ids, padding_mask=padding_mask, causal_mask=causal_mask)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Step 5-6: GPT(Decoder-only) 모델 학습")
    print("=" * 60)

    # 환경 변수로 하이퍼파라미터 오버라이드 (run_pipeline.sh 대화형 입력 연동)
    config.EPOCHS = int(os.environ.get("EPOCHS", config.EPOCHS))
    config.DROPOUT_RATE = float(os.environ.get("DROPOUT", config.DROPOUT_RATE))
    config.LEARNING_RATE = float(os.environ.get("LR", config.LEARNING_RATE))
    config.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", config.BATCH_SIZE))
    config.MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", config.MAX_SEQ_LENGTH))

    print(f"하이퍼파라미터: EPOCHS={config.EPOCHS} | DROPOUT={config.DROPOUT_RATE} | "
          f"LR={config.LEARNING_RATE} | BATCH={config.BATCH_SIZE} | SEQ={config.MAX_SEQ_LENGTH}")

    # 데이터 로드
    if not os.path.exists(config.AUGMENTED_CSV):
        print(f"Error: Augmented data not found")
        print("Please run 03_augmentation.py first")
        return

    print(f"\nLoading augmented data from {config.AUGMENTED_CSV}...")
    df = pd.read_csv(config.AUGMENTED_CSV)
    print(f"Loaded {len(df)} sentences")

    # 원본 데이터 크기 (증강 전)
    original_data_size = None
    if os.path.exists(config.CLEANED_CSV):
        original_data_size = len(pd.read_csv(config.CLEANED_CSV))
    augmented_data_size = len(df)

    # 어휘 로드
    if not os.path.exists(config.CORPUS_PKL):
        print(f"Error: Corpus not found")
        print("Please run 02_build_corpus.py first")
        return

    print(f"Loading vocabulary from corpus...")
    vocab, tokenizer_name = create_vocab_from_corpus(config.CORPUS_PKL)
    print(f"Vocabulary size: {len(vocab)}  (tokenizer: {tokenizer_name})")

    # <sep> 토큰이 vocab에 없으면 추가
    if '<sep>' not in vocab:
        sep_idx = max(vocab.values()) + 1
        vocab['<sep>'] = sep_idx
        print(f"<sep> token added to vocab (idx={sep_idx}), new vocab size: {len(vocab)}")

    # 실험 추적기 초기화
    tracker = ExperimentTracker()

    # 데이터셋 및 데이터로더 생성
    print(f"\nCreating dataset...")
    dataset = GPTDataset(df, vocab)

    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    batch_size = config.BATCH_SIZE

    # RTX 4090에서 메모리 사용량 추정
    num_params = (config.VOCAB_SIZE * config.TRANSFORMER_D_MODEL +
                  config.MAX_SEQ_LENGTH * config.TRANSFORMER_D_MODEL * config.TRANSFORMER_NHEAD * batch_size)
    estimated_memory_mb = (num_params * 4) / (1024 * 1024)
    print(f"\nRTX 4090 최적화 설정:")
    print(f"Batch size: {batch_size}")
    print(f"Estimated memory per batch: ~{estimated_memory_mb:.0f}MB")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 모델 생성
    print(f"\nCreating GPT model...")
    model = GPTModel(vocab_size=len(vocab)).to(device)

    # 이어서 학습 (resume)
    if config.RESUME_TRAINING and os.path.exists(config.MODEL_SAVE_PATH):
        print(f"기존 모델 발견: {config.MODEL_SAVE_PATH} -> 이어서 학습합니다.")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    elif config.RESUME_TRAINING:
        print(f"기존 모델 없음 -> 새로 학습을 시작합니다.")
    else:
        print(f"RESUME_TRAINING=False -> 새로 학습을 시작합니다.")

    # 모델 파라미터 수
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 옵티마이저와 loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 토큰(0) 무시
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='linear'
    )

    # 학습 루프
    print(f"\nStarting training...")
    print(f"Epochs: {config.EPOCHS}")

    best_val_loss = float('inf')
    best_epoch = 1
    train_losses = []
    val_losses = []

    for epoch in range(config.EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{config.EPOCHS}]")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Best model saved (val_loss: {val_loss:.4f})")

    # 결과 저장
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'vocab_size': len(vocab),
        'model_params': num_params,
        'batch_size': batch_size,
        'epochs': config.EPOCHS,
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(config.MODEL_PATH, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 토크나이저 저장
    tokenizer = Tokenizer(vocab)
    tokenizer.save(config.TOKENIZER_PATH)

    # 실험 기록 저장 및 비교 출력
    tracker.save(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        tokenizer_name=tokenizer_name,
        original_data_size=original_data_size,
        augmented_data_size=augmented_data_size,
    )

    print(f"\nTraining completed successfully!")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
