import sys
sys.path.insert(0, '/workspace/test/NLP/NLP05')

import os, sys, json, random
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.append("/workspace/NLP/NLP05")
import config as cfg

random.seed(cfg.RANDOM_SEED)

# ── 1. 단일 모델 추론 함수 ───────────────────────────────
def load_model(model_path, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, decode_cfg, device, max_new_tokens=64):
    """
    config.py의 DECODING_CONFIGS 딕셔너리를 그대로 받아서 실행
    → greedy / beam / top-k / top-p 모두 이 함수 하나로 처리
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **decode_cfg   # DECODING_CONFIGS 값 언패킹
        )
    # 프롬프트 부분 제거, 생성된 응답만 반환
    gen_ids = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── 2. BLEU / ROUGE 계산 ─────────────────────────────────
def compute_bleu(reference, hypothesis):
    """
    BLEU: n-gram 정밀도 기반 유사도
    smoothing: 짧은 문장에서 0이 되는 문제 방지
    """
    ref_tokens  = list(reference)    # 한국어: 음절 단위 토큰화
    hyp_tokens  = list(hypothesis)
    smoother    = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)

def compute_rouge(reference, hypothesis):
    """
    ROUGE-L: 최장 공통 부분 수열(LCS) 기반 재현율
    한국어 텍스트 비교에 BLEU보다 안정적
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score  = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


# ── 3. 3단계 모델 비교 ───────────────────────────────────
def compare_models(prompts, device):
    """
    루브릭①: baseline vs SFT
    루브릭②: SFT vs PPO (RM 포함)
    """
    results = []

    print("모델 로드 중...")
    baseline_model, baseline_tok = load_model(cfg.BASE_MODEL_NAME, device)
    sft_model,      sft_tok      = load_model(cfg.SFT_SAVE_PATH,   device)
    ppo_model,      ppo_tok      = load_model(cfg.PPO_SAVE_PATH,   device)

    # greedy decoding으로 공정 비교
    decode_cfg = cfg.DECODING_CONFIGS["greedy"]

    for prompt in prompts:
        baseline_out = generate(baseline_model, baseline_tok, prompt, decode_cfg, device)
        sft_out      = generate(sft_model,      sft_tok,      prompt, decode_cfg, device)
        ppo_out      = generate(ppo_model,      ppo_tok,      prompt, decode_cfg, device)

        results.append({
            "prompt":   prompt,
            "baseline": baseline_out,
            "sft":      sft_out,
            "ppo":      ppo_out,
        })
        print(f"\n[프롬프트] {prompt}")
        print(f"  baseline: {baseline_out}")
        print(f"  SFT     : {sft_out}")
        print(f"  PPO     : {ppo_out}")

    return results


# ── 4. Decoding 실험 ─────────────────────────────────────
def decoding_experiment(prompts, device):
    """
    루브릭③: PPO 모델에 다양한 decoding 전략 적용
    greedy / beam_2 / beam_4 / topk_10 / topk_50 / topp_09
    """
    model, tokenizer = load_model(cfg.PPO_SAVE_PATH, device)
    results = []

    for prompt in prompts:
        row = {"prompt": prompt}
        for name, decode_cfg in cfg.DECODING_CONFIGS.items():
            out = generate(model, tokenizer, prompt, decode_cfg, device)
            row[name] = out
        results.append(row)
        print(f"\n[프롬프트] {prompt}")
        for name in cfg.DECODING_CONFIGS:
            print(f"  {name:10s}: {results[-1][name]}")

    return results


# ── 5. 정량 평가 (BLEU + ROUGE) ──────────────────────────
def quantitative_eval(model_outputs, references):
    """
    model_outputs: [{"baseline": ..., "sft": ..., "ppo": ...}, ...]
    references:    정답 응답 리스트 (SFT 데이터의 completion 활용)
    """
    scores = {"baseline": [], "sft": [], "ppo": []}

    for out, ref in zip(model_outputs, references):
        for stage in ["baseline", "sft", "ppo"]:
            bleu  = compute_bleu(ref, out[stage])
            rouge = compute_rouge(ref, out[stage])
            scores[stage].append({"bleu": bleu, "rouge_l": rouge})

    # 평균 계산
    summary = {}
    for stage in scores:
        bleu_avg  = sum(s["bleu"]    for s in scores[stage]) / len(scores[stage])
        rouge_avg = sum(s["rouge_l"] for s in scores[stage]) / len(scores[stage])
        summary[stage] = {"avg_bleu": bleu_avg, "avg_rouge_l": rouge_avg}
        print(f"[{stage:8s}] BLEU={bleu_avg:.4f} | ROUGE-L={rouge_avg:.4f}")

    return summary


# ── 6. LLM-as-a-Judge 프롬프트 생성 ─────────────────────
def build_judge_prompt(question, response_a, response_b):
    """
    강의에서 소개된 LLM-as-a-Judge 방식
    BLEU의 한계(n-gram만 측정)를 보완하는 자동 평가
    Claude/ChatGPT API로 전송할 프롬프트 생성
    """
    return f"""다음 질문에 대한 두 응답을 비교하여 평가해주세요.

[질문]
{question}

[응답 A]
{response_a}

[응답 B]
{response_b}

다음 기준으로 각 응답을 1~5점으로 평가하고, 더 나은 응답을 선택해주세요:
1. 유창성: 자연스럽고 읽기 쉬운가?
2. 관련성: 질문에 적절히 답하고 있는가?
3. 정보성: 유용한 정보를 담고 있는가?

출력 형식:
응답A 점수: [유창성/관련성/정보성]
응답B 점수: [유창성/관련성/정보성]
더 나은 응답: [A 또는 B]
이유: [한 문장 설명]
"""


# ── 7. 메인 ─────────────────────────────────────────────
def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 테스트 프롬프트 (SFT 데이터에서 샘플링)
    test_prompts = []
    with open(cfg.SFT_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = random.sample(raw, min(20, len(raw)))
    for item in samples:
        test_prompts.append(item["prompt"])

    # 정답 참조 응답 (BLEU/ROUGE용)
    references = []
    for item in samples:
        references.append(item["completion"])

    print(f"\n{'='*60}")
    print("① 3단계 모델 비교 (baseline vs SFT vs PPO)")
    print(f"{'='*60}")
    model_outputs = compare_models(test_prompts, device)

    print(f"\n{'='*60}")
    print("② 정량 평가 (BLEU + ROUGE-L)")
    print(f"{'='*60}")
    quant_summary = quantitative_eval(model_outputs, references)

    print(f"\n{'='*60}")
    print("③ Decoding 실험 (PPO 모델)")
    print(f"{'='*60}")
    decoding_results = decoding_experiment(test_prompts[:5], device)

    print(f"\n{'='*60}")
    print("④ LLM-as-a-Judge 프롬프트 샘플 (SFT vs PPO)")
    print(f"{'='*60}")
    for i, (out, prompt) in enumerate(zip(model_outputs[:3], test_prompts[:3])):
        print(f"\n--- 샘플 {i+1} ---")
        print(build_judge_prompt(prompt, out["sft"], out["ppo"]))

    # 결과 저장
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(model_outputs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "quant_summary.json"), "w", encoding="utf-8") as f:
        json.dump(quant_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "decoding_results.json"), "w", encoding="utf-8") as f:
        json.dump(decoding_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
