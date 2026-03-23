# NLP04 - mini BERT Pretrain

vocab_size=8000, 전체 파라미터 ~1M의 mini BERT를 PyTorch로 구현하고 MLM + NSP 태스크로 10 Epoch 사전학습합니다.

## 모델 사양

| 항목 | 값 |
|------|-----|
| vocab_size | 8,000 |
| d_model | 128 |
| num_heads | 4 |
| num_layers | 3 |
| d_ff | 256 |
| max_seq_len | 128 |
| 총 파라미터 | ~1M |

## 폴더 구조

```
NLP04/
├── data/
│   ├── raw/            # kowiki dump, corpus.txt
│   └── processed/      # spm.model, spm.vocab, memmap 데이터
├── models/             # 체크포인트, 학습 로그, 그래프
├── scripts/
│   ├── 01_build_tokenizer.py   # SentencePiece BPE 학습
│   ├── 02_preprocess_data.py   # MLM + NSP 전처리 → memmap 저장
│   └── 03_pretrain.py          # mini BERT 구현 및 학습
├── config.json
└── README.md
```

## 실행 순서

```bash
# 1. 토크나이저 학습 (kowiki dump 다운로드 포함)
python scripts/01_build_tokenizer.py

# 2. 전처리 (MLM + NSP)
python scripts/02_preprocess_data.py
# 샘플 수 제한 시: python scripts/02_preprocess_data.py --max_samples 100000

# 3. 사전학습
python scripts/03_pretrain.py
```

## 핵심 구현

### MLM (Masked Language Modeling)
- 전체 토큰의 15% 선택
- 선택된 토큰 중 80% → [MASK], 10% → 랜덤 토큰, 10% → 원본 유지
- `mlm_labels`: 마스킹 위치만 실제 ID, 나머지는 -100 (CrossEntropyLoss ignore)

### NSP (Next Sentence Prediction)
- 50% IsNext(1): 실제 연속 문장 쌍
- 50% NotNext(0): 랜덤 두 번째 문장
- 포맷: `[CLS] 문장A [SEP] 문장B [SEP]`
- segment_ids: 문장A=0, 문장B=1

### 학습
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: WarmupLinearSchedule (warmup=전체 step의 10%)
- Gradient clipping: max_norm=1.0
- total_loss = mlm_loss + nsp_loss

## 출력 결과

- `models/bert_best.pt`: 최적 체크포인트
- `models/bert_final.pt`: 최종 체크포인트
- `models/epoch_XX_log.json`: epoch별 loss/accuracy
- `models/train_history.json`: 전체 학습 히스토리
- `models/pretrain_loss.png`: loss 시각화 그래프

## 의존 패키지

```bash
pip install torch sentencepiece numpy tqdm matplotlib --break-system-packages
```