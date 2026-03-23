# chatgptbot - GPT(Decoder-only) 한국어 챗봇

## 개요

songys/Chatbot_data(한국어 챗봇 Q&A 데이터)를 사용하여 GPT(Decoder-only Transformer) 모델을 학습하고, 대화형 추론을 수행하는 프로젝트입니다. RTX 4090(24GB VRAM) 환경에 최적화되어 있습니다.

## 모델 구조

- **아키텍처**: Decoder-only Transformer (GPT)
- **학습 방식**: Next Token Prediction (`<start> Q <sep> A <end>` 단일 시퀀스)
- **위치 인코딩**: 학습 가능한 Positional Embedding (`nn.Embedding`)
- **주요 설정**: d_model=768, nhead=12, 6 layers, feedforward=3072

## 폴더 구조

```
chatgptbot/
├── config.py                  # 하이퍼파라미터 및 경로 설정
├── run_pipeline.sh            # 전체 파이프라인 실행 (대화형 하이퍼파라미터 입력)
├── setup.sh                   # 환경 설정 (Java, MeCab, Python 패키지 설치)
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── ChatbotData.csv            # 원본 데이터 (songys/Chatbot_data)
│   └── processed/
│       ├── cleaned_data.csv           # 전처리된 데이터
│       ├── augmented_data.csv         # Word2Vec 기반 증강 데이터 (3배)
│       └── corpus.pkl                 # 어휘 사전 + 토큰화 결과
│
├── scripts/
│   ├── 01_preprocess.py               # Step 1-2: 정규식 정제, 결측값/중복 제거
│   ├── 02_build_corpus.py             # Step 3: 형태소 분석(Mecab/Okt) + 어휘 사전 구축
│   ├── 03_augmentation.py             # Step 4: Word2Vec 학습 + Lexical Substitution 증강
│   ├── 04_train.py                    # Step 5-6: GPT 모델 정의 및 학습
│   ├── 05_inference.py                # 대화형 챗봇 추론 모드
│   └── experiment_tracker.py          # 실험 자동 추적 (CSV/JSON 기록 + 시각화)
│
├── models/
│   ├── tokenizer.pkl                  # 저장된 토크나이저
│   ├── training_results.json          # 에폭별 Train/Val Loss
│   ├── experiments_log.csv            # 실험 기록 (CSV)
│   └── experiments_log.json           # 실험 기록 (JSON)
│
├── notebooks/
│   ├── gpt_quest1.ipynb               # GPT 퀘스트 1: 아키텍처 비교 + 데이터/모델/생성 실습
│   ├── gpt_quest2.ipynb               # GPT 퀘스트 2
│   ├── gpt_quest3.ipynb               # GPT 퀘스트 3
│   └── analysis.ipynb                 # 데이터 분석 노트북
│
├── training_loss.png                  # Train/Val Loss 곡선
├── length_distribution.png            # 문장 길이 분포
└── top_words.png                      # 상위 빈출 단어
```

## 파이프라인

```
run_pipeline.sh 실행 시 순서:

1. 하이퍼파라미터 대화형 입력 (에폭, 드롭아웃, LR, 배치 등)
2. 01_preprocess.py    → 데이터 정제 (정규식, 결측값/중복 제거)
3. 02_build_corpus.py  → 형태소 분석 + 어휘 사전 구축
4. 03_augmentation.py  → Word2Vec 유사어 치환으로 3배 증강
5. 04_train.py         → GPT 모델 학습 (Train/Val split 8:2)
```

## 실행 방법

```bash
# 1. 환경 설정 (최초 1회)
bash setup.sh

# 2. 전체 파이프라인 실행
bash run_pipeline.sh

# 3. 대화형 추론
python scripts/05_inference.py
```

## 주요 기술

- **형태소 분석**: Mecab → Okt → whitespace (순차 fallback)
- **데이터 증강**: Word2Vec(Skip-gram) 학습 후 Lexical Substitution으로 질문/답변 각각 치환 (3배)
- **실험 추적**: `experiment_tracker.py`가 매 학습마다 하이퍼파라미터, 데이터 설정, 결과를 자동 기록하고 이전 실험과 비교 출력
