"""
Step 7: Retrieval 기반 대화형 추론 스크립트
- cleaned_data.csv 전체를 인코더로 임베딩 → 인덱스 구축
- 사용자 입력을 같은 인코더로 임베딩 → 코사인 유사도 Top-3 검색
- 1위 답변을 챗봇 응답으로, 2~3위는 유사 질문 참고로 출력
- 실행: python scripts/05_inference.py
"""

import os
import sys
import re
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

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

    def encode(self, src, src_key_padding_mask=None):
        """인코더만 실행해 문장 표현 반환 (batch, seq, d_model)"""
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        src_emb = src_emb + self.positional_encoding[:, :src.size(1), :].to(src.device)
        src_emb = self.dropout(src_emb)
        return self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

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
# 토크나이저
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
        print("✓ Mecab 토크나이저 로드 (폴백)")
        return tok
    except Exception:
        pass
    print("⚠ whitespace 토크나이저 사용")
    return None


def _tokenize(sentence, tokenizer):
    try:
        return tokenizer.morphs(sentence) if tokenizer else sentence.split()
    except Exception:
        return sentence.split()


def _preprocess(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣.!?]', '', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# 인코딩 유틸
# ──────────────────────────────────────────────────────────────

def _encode_sentence(sentence, tokenizer, word2idx, max_len):
    """문장 → 패딩된 토큰 ID 텐서 (1, max_len)"""
    pad_id = word2idx.get('<pad>', 0)
    unk_id = word2idx.get('<unk>', 1)
    tokens = _tokenize(_preprocess(sentence), tokenizer)
    ids = [word2idx.get(t, unk_id) for t in tokens][:max_len]
    ids += [pad_id] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def _get_embedding(sentence, model, tokenizer, word2idx, device, max_len):
    """
    문장 → 인코더 출력의 평균 풀링 벡터 (d_model,)
    패딩 토큰은 평균에서 제외
    """
    pad_id = word2idx.get('<pad>', 0)
    src = _encode_sentence(sentence, tokenizer, word2idx, max_len).to(device)
    src_mask = (src == pad_id)

    with torch.no_grad():
        enc_out = model.encode(src, src_key_padding_mask=src_mask)  # (1, seq, d_model)

    # 패딩 제외 평균 풀링
    non_pad = (~src_mask[0]).float().unsqueeze(-1)  # (seq, 1)
    vec = (enc_out[0] * non_pad).sum(0) / non_pad.sum().clamp(min=1)
    return F.normalize(vec, dim=0)  # L2 정규화


# ──────────────────────────────────────────────────────────────
# 인덱스 구축
# ──────────────────────────────────────────────────────────────

def build_index(df, model, tokenizer, word2idx, device, max_len, batch_size=256):
    """
    cleaned_data.csv의 모든 질문을 인코더로 임베딩해 행렬로 저장

    Returns:
        emb_matrix: (N, d_model) float32 텐서 (CPU)
        questions:  list[str]
        answers:    list[str]
    """
    questions = df['question'].tolist()
    answers   = df['answer'].tolist()
    pad_id    = word2idx.get('<pad>', 0)
    unk_id    = word2idx.get('<unk>', 1)

    all_vecs = []
    model.eval()

    print(f"인덱스 구축 중... ({len(questions)}개 질문)")
    for start in range(0, len(questions), batch_size):
        batch_q = questions[start:start + batch_size]

        # 배치 토큰화 + 패딩
        batch_ids = []
        for q in batch_q:
            tokens = _tokenize(_preprocess(q), tokenizer)
            ids = [word2idx.get(t, unk_id) for t in tokens][:max_len]
            ids += [pad_id] * (max_len - len(ids))
            batch_ids.append(ids)

        src = torch.tensor(batch_ids, dtype=torch.long).to(device)
        src_mask = (src == pad_id)

        with torch.no_grad():
            enc_out = model.encode(src, src_key_padding_mask=src_mask)  # (B, seq, d)

        non_pad = (~src_mask).float().unsqueeze(-1)           # (B, seq, 1)
        vecs = (enc_out * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)  # (B, d)
        vecs = F.normalize(vecs, dim=1)
        all_vecs.append(vecs.cpu())

    emb_matrix = torch.cat(all_vecs, dim=0)  # (N, d_model)
    print(f"✓ 인덱스 구축 완료 (shape: {emb_matrix.shape})")
    return emb_matrix, questions, answers


# ──────────────────────────────────────────────────────────────
# 검색
# ──────────────────────────────────────────────────────────────

def retrieve(query, model, tokenizer, word2idx, device, max_len,
             emb_matrix, questions, answers, top_k=3):
    """
    query → 코사인 유사도 기반 Top-k (score, question, answer) 리스트
    """
    q_vec = _get_embedding(query, model, tokenizer, word2idx, device, max_len)  # (d,)
    scores = emb_matrix.to(device) @ q_vec  # (N,) — L2 정규화 후 내적 = 코사인 유사도

    top_scores, top_idx = scores.topk(top_k)
    results = []
    for score, idx in zip(top_scores.cpu().tolist(), top_idx.cpu().tolist()):
        results.append((score, questions[idx], answers[idx]))
    return results


# ──────────────────────────────────────────────────────────────
# 메인 — 대화형 루프
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("한국어 챗봇 추론 (Retrieval 방식)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 파일 존재 확인 ──
    for path, label in [
        (config.TOKENIZER_PATH, "tokenizer.pkl"),
        (config.MODEL_SAVE_PATH, "chatbot_model.pt"),
        (config.CLEANED_CSV,     "cleaned_data.csv"),
    ]:
        if not os.path.exists(path):
            print(f"✗ {label} 없음: {path}")
            print("  먼저 run_pipeline.sh를 실행하세요.")
            return

    # ── tokenizer.pkl 로드 ──
    with open(config.TOKENIZER_PATH, 'rb') as f:
        tok_data = pickle.load(f)
    word2idx   = tok_data['word2idx']
    idx2word   = tok_data['idx2word']
    vocab_size = len(word2idx)
    print(f"✓ tokenizer.pkl 로드 완료 (vocab: {vocab_size})")

    # ── 모델 로드 ──
    model = TransformerChatbot(vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"✓ chatbot_model.pt 로드 완료")

    # ── 토크나이저 ──
    tokenizer = _init_tokenizer()

    # ── cleaned_data.csv 로드 + 인덱스 구축 ──
    df = pd.read_csv(config.CLEANED_CSV)
    print(f"✓ cleaned_data.csv 로드 완료 ({len(df)}개 행)")

    max_len    = config.MAX_SEQ_LENGTH
    emb_matrix, questions, answers = build_index(
        df, model, tokenizer, word2idx, device, max_len)

    print("\n대화를 시작합니다. 종료하려면 'q'를 입력하세요.")
    print("-" * 60)

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

        results = retrieve(
            user_input, model, tokenizer, word2idx, device, max_len,
            emb_matrix, questions, answers, top_k=3
        )

        # 1위: 챗봇 답변
        best_score, best_q, best_a = results[0]
        print(f"\n챗봇: {best_a}")
        print(f"유사도: {best_score * 100:.1f}%  (매칭 질문: {best_q})")

        # 2~3위: 유사 질문 참고
        if len(results) > 1:
            print("\n[유사 질문 참고]")
            for rank, (score, q, a) in enumerate(results[1:], start=2):
                print(f"  {rank}위 ({score * 100:.1f}%)  Q: {q}")
                print(f"          A: {a}")

        print("-" * 60)


if __name__ == "__main__":
    main()
