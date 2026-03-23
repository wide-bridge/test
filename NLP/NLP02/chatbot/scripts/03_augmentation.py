"""
Step 4: 코퍼스 기반 Word2Vec 학습 + Lexical Substitution 데이터 증강
- ko.bin 없이 현재 데이터(질문+답변)로 Word2Vec 직접 학습
- lexical_sub()로 각 토큰을 0.3 확률로 유사어 치환
- 증강 구성: 원본 / 질문증강+원본답변 / 원본질문+답변증강 → 3배
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_tokenizer():
    """Okt 토크나이저 반환 (Mecab 실패 시 Okt, 그 다음 whitespace)"""
    try:
        from konlpy.tag import Mecab
        tok = Mecab()
        tok.morphs("테스트")
        print("✓ Mecab 토크나이저 사용")
        return tok
    except Exception:
        pass
    try:
        from konlpy.tag import Okt
        tok = Okt()
        print("✓ Okt 토크나이저 사용")
        return tok
    except Exception:
        pass
    print("⚠ whitespace 토크나이저 사용 (형태소 분석 불가)")
    return None


def tokenize(sentence, tokenizer):
    """문장 → 형태소 리스트"""
    try:
        if tokenizer is None:
            return sentence.split()
        return tokenizer.morphs(sentence)
    except Exception:
        return sentence.split()


def build_w2v(sentences_tokenized):
    """
    토큰화된 문장 리스트로 Word2Vec 학습

    Args:
        sentences_tokenized: list of list of str

    Returns:
        gensim Word2Vec 모델
    """
    print("Word2Vec 학습 중...")
    model = Word2Vec(
        sentences=sentences_tokenized,
        vector_size=200,
        window=5,
        min_count=1,
        sg=1,        # Skip-gram
        epochs=20,
        workers=4,
    )
    print(f"✓ Word2Vec 학습 완료 (vocab: {len(model.wv)})")
    return model


def lexical_sub(tokens, w2v_model, vocab, replace_prob=0.3):
    """
    각 토큰을 replace_prob 확률로 Word2Vec 유사어로 치환

    Args:
        tokens (list[str]): 형태소 리스트
        w2v_model: 학습된 Word2Vec 모델
        vocab (set): 유효 어휘 집합 (word2idx 키)
        replace_prob (float): 치환 확률

    Returns:
        list[str]: 치환된 형태소 리스트
    """
    result = []
    for token in tokens:
        if np.random.random() < replace_prob and token in w2v_model.wv:
            candidates = [
                w for w, _ in w2v_model.wv.most_similar(token, topn=10)
                if w in vocab and w != token
            ]
            result.append(candidates[0] if candidates else token)
        else:
            result.append(token)
    return result


def augment_sentence(sentence, tokenizer, w2v_model, vocab):
    """문장 → 증강된 문장 (토큰화 → 치환 → 재결합)"""
    tokens = tokenize(sentence, tokenizer)
    aug_tokens = lexical_sub(tokens, w2v_model, vocab)
    return " ".join(aug_tokens)


def main():
    print("=" * 60)
    print("Step 4: 코퍼스 기반 Word2Vec 증강")
    print("=" * 60)

    # 전처리 데이터 로드
    if not os.path.exists(config.CLEANED_CSV):
        print(f"✗ Error: {config.CLEANED_CSV} 없음. 01_preprocess.py를 먼저 실행하세요.")
        return

    print(f"\n데이터 로드: {config.CLEANED_CSV}")
    df = pd.read_csv(config.CLEANED_CSV)
    print(f"✓ {len(df)}개 문장 로드")

    # 증강 스킵 옵션
    if not getattr(config, 'USE_AUGMENTATION', True):
        print("⚠ USE_AUGMENTATION=False → 증강 없이 원본 저장")
        os.makedirs(os.path.dirname(config.AUGMENTED_CSV), exist_ok=True)
        df.to_csv(config.AUGMENTED_CSV, index=False, encoding='utf-8')
        print(f"✓ {config.AUGMENTED_CSV} 저장 완료")
        return

    # 코퍼스(vocab) 로드
    if not os.path.exists(config.CORPUS_PKL):
        print(f"✗ Error: {config.CORPUS_PKL} 없음. 02_build_corpus.py를 먼저 실행하세요.")
        return

    with open(config.CORPUS_PKL, 'rb') as f:
        corpus_data = pickle.load(f)
    vocab = set(corpus_data['word2idx'].keys())
    print(f"✓ 어휘 로드 완료 ({len(vocab)}개)")

    # 토크나이저 초기화
    tokenizer = get_tokenizer()

    # 전체 문장 토큰화 (Word2Vec 학습용)
    print("\n전체 문장 토큰화 중...")
    all_tokenized = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        all_tokenized.append(tokenize(row['question'], tokenizer))
        all_tokenized.append(tokenize(row['answer'], tokenizer))

    # Word2Vec 학습
    w2v_model = build_w2v(all_tokenized)

    # 3배 증강
    # - 원본
    # - 증강본 1: 질문증강 + 원본답변
    # - 증강본 2: 원본질문 + 답변증강
    print("\n증강 데이터 생성 중...")

    aug1_questions, aug1_answers = [], []
    aug2_questions, aug2_answers = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        aug1_questions.append(augment_sentence(row['question'], tokenizer, w2v_model, vocab))
        aug1_answers.append(row['answer'])

        aug2_questions.append(row['question'])
        aug2_answers.append(augment_sentence(row['answer'], tokenizer, w2v_model, vocab))

    aug1_df = pd.DataFrame({'question': aug1_questions, 'answer': aug1_answers})
    aug2_df = pd.DataFrame({'question': aug2_questions, 'answer': aug2_answers})

    augmented_df = pd.concat([df, aug1_df, aug2_df], ignore_index=True)

    print(f"\n원본 크기:  {len(df)}")
    print(f"증강 후 크기: {len(augmented_df)}  ({len(augmented_df) / len(df):.1f}x)")

    # 샘플 확인
    print("\n[샘플 — 원본 vs 증강]")
    sample = df.iloc[0]
    aug_q = augment_sentence(sample['question'], tokenizer, w2v_model, vocab)
    aug_a = augment_sentence(sample['answer'], tokenizer, w2v_model, vocab)
    print(f"  원본 질문: {sample['question']}")
    print(f"  증강 질문: {aug_q}")
    print(f"  원본 답변: {sample['answer']}")
    print(f"  증강 답변: {aug_a}")

    # 저장
    os.makedirs(os.path.dirname(config.AUGMENTED_CSV), exist_ok=True)
    augmented_df.to_csv(config.AUGMENTED_CSV, index=False, encoding='utf-8')
    print(f"\n✓ 증강 데이터 저장: {config.AUGMENTED_CSV}")
    print("✓ 데이터 증강 완료!")


if __name__ == "__main__":
    main()
