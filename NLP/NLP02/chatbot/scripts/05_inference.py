"""
Step 7: 대화형 추론 스크립트
- 학습된 chatbot_model.pt + tokenizer.pkl 자동 로드
- Okt 토크나이저 사용 (Mecab 실패 시 폴백)
- 답변과 함께 신뢰도 점수(top-1 softmax 확률 기하평균) 출력
- 실행: python scripts/05_inference.py
"""

import os
import sys
import re
import pickle

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ──────────────────────────────────────────────────────────────
# 모델 정의 (04_train.py와 동일 구조)
# ──────────────────────────────────────────────────────────────

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size,
                 d_model=config.TRANSFORMER_D_MODEL,
                 nhead=config.TRANSFORMER_NHEAD,
                 num_layers=config.TRANSFORMER_NUM_LAYERS,
                 dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
                 dropout=config.DROPOUT_RATE):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = self._get_positional_encoding(d_model, config.MAX_SEQ_LENGTH)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _get_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_mask=None):
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * np.sqrt(self.d_model)
        src_emb = src_emb + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        enc_out = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.transformer_decoder(
            tgt_emb, enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(dec_out)


# ──────────────────────────────────────────────────────────────
# 토크나이저 초기화 (Okt 우선, Mecab 폴백)
# ──────────────────────────────────────────────────────────────

def _init_tokenizer():
    try:
        from konlpy.tag import Okt
        tok = Okt()
        tok.morphs("테스트")
        print("✓ Okt 토크나이저 로드")
        return tok
    except Exception:
        pass
    try:
        from konlpy.tag import Mecab
        tok = Mecab()
        print("✓ Mecab 토크나이저 로드")
        return tok
    except Exception:
        pass
    print("⚠ whitespace 토크나이저 사용 (형태소 분석 불가)")
    return None


def _tokenize(sentence, tokenizer):
    try:
        if tokenizer is None:
            return sentence.split()
        return tokenizer.morphs(sentence)
    except Exception:
        return sentence.split()


# ──────────────────────────────────────────────────────────────
# 전처리
# ──────────────────────────────────────────────────────────────

def _preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣.!?]', '', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# 추론
# ──────────────────────────────────────────────────────────────

def infer(question, model, word2idx, idx2word, tokenizer, device,
          max_len=None, temperature=0.8, top_k=10, no_repeat_ngram=4):
    """
    question → (answer, confidence)

    디코딩 전략:
      - temperature: 확률 분포 조절 (< 1.0 이면 더 집중적, > 1.0 이면 더 다양)
      - top_k sampling: 상위 k개 토큰 중 확률 비례 랜덤 선택
      - 반복 억제: 직전 no_repeat_ngram개 토큰과 동일한 토큰 확률 0 처리
    confidence: 각 스텝 선택 토큰의 softmax 확률 기하평균
    """
    if max_len is None:
        max_len = config.MAX_SEQ_LENGTH

    pad_id   = word2idx.get('<pad>', 0)
    unk_id   = word2idx.get('<unk>', 1)
    start_id = word2idx.get('<start>', 2)
    end_id   = word2idx.get('<end>', 3)

    text = _preprocess(question)
    tokens = _tokenize(text, tokenizer)

    # 인코더 입력 인코딩
    ids = [word2idx.get(t, unk_id) for t in tokens][:max_len]
    ids += [pad_id] * (max_len - len(ids))
    src = torch.tensor([ids], dtype=torch.long).to(device)
    src_mask = (src == pad_id)

    model.eval()
    with torch.no_grad():
        tgt = torch.tensor([[start_id]], dtype=torch.long).to(device)
        step_probs = []

        for _ in range(max_len - 1):
            tgt_len = tgt.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
            tgt_pad_mask = (tgt == pad_id)

            out = model(src, tgt,
                        src_key_padding_mask=src_mask,
                        tgt_key_padding_mask=tgt_pad_mask,
                        tgt_mask=causal_mask)

            logits = out[0, -1, :].clone()  # (vocab_size,)

            # 특수 토큰 억제
            logits[start_id] = -1e9
            logits[pad_id]   = -1e9

            # 반복 억제: 직전 no_repeat_ngram개 토큰과 같으면 확률 0
            recent = tgt[0, -no_repeat_ngram:].tolist()
            for rid in set(recent):
                logits[rid] = -1e9

            # temperature 적용
            logits = logits / temperature

            # top-k masking: 상위 k개 제외 나머지 -inf
            top_k_vals, _ = torch.topk(logits, top_k)
            threshold = top_k_vals[-1]
            logits[logits < threshold] = -1e9

            probs = torch.softmax(logits, dim=-1)

            # 확률 비례 샘플링
            next_id = torch.multinomial(probs, num_samples=1).item()
            step_probs.append(probs[next_id].item())

            # EOS 도달 시 종료
            if next_id == end_id:
                break

            tgt = torch.cat(
                [tgt, torch.tensor([[next_id]], dtype=torch.long).to(device)],
                dim=1
            )

    # 토큰 → 문자열
    resp_ids = tgt[0].cpu().tolist()[1:]  # <start> 제거
    resp_tokens = [idx2word.get(i, '') for i in resp_ids]
    if '<end>' in resp_tokens:
        resp_tokens = resp_tokens[:resp_tokens.index('<end>')]
    answer = ''.join(resp_tokens).strip()

    # 신뢰도: 기하평균 (log 합산 후 exp)
    if step_probs:
        confidence = float(np.exp(np.mean(np.log(np.clip(step_probs, 1e-9, 1.0)))))
    else:
        confidence = 0.0

    return answer, confidence


# ──────────────────────────────────────────────────────────────
# 메인 — 대화형 루프
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("한국어 챗봇 추론")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 토크나이저 로드
    if not os.path.exists(config.TOKENIZER_PATH):
        print(f"✗ tokenizer.pkl 없음: {config.TOKENIZER_PATH}")
        print("  먼저 run_pipeline.sh 또는 04_train.py를 실행하세요.")
        return

    with open(config.TOKENIZER_PATH, 'rb') as f:
        tok_data = pickle.load(f)
    word2idx = tok_data['word2idx']
    idx2word = tok_data['idx2word']
    vocab_size = len(word2idx)
    print(f"✓ tokenizer.pkl 로드 완료 (vocab: {vocab_size})")

    # 모델 로드
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"✗ chatbot_model.pt 없음: {config.MODEL_SAVE_PATH}")
        print("  먼저 run_pipeline.sh 또는 04_train.py를 실행하세요.")
        return

    model = TransformerChatbot(vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print(f"✓ chatbot_model.pt 로드 완료")

    # 토크나이저 초기화
    tokenizer = _init_tokenizer()

    print("\n대화를 시작합니다. 종료하려면 'q'를 입력하세요.")
    print("-" * 55)

    while True:
        try:
            user_input = input("질문을 입력하세요 (종료: q): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if user_input.lower() == 'q':
            print("종료합니다.")
            break

        if not user_input:
            continue

        answer, confidence = infer(
            user_input, model, word2idx, idx2word, tokenizer, device
        )

        print(f"챗봇: {answer}")
        print(f"신뢰도: {confidence * 100:.1f}%")
        print("-" * 55)


if __name__ == "__main__":
    main()
