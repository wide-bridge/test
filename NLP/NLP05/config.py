# =============================================================================
# NLP05 KoChatGPT Config
# NLP03/config.py 구조 계승 + HuggingFace 경로 추가
# =============================================================================

# --- 경로 (RunPod 절대경로 기준) ---
BASE_DIR            = "/workspace"
DATA_DIR            = "/workspace/KoChatGPT/data_kochatgpt"
NLP05_DIR           = "/workspace/NLP/NLP05"
RESULTS_DIR         = "/workspace/NLP/NLP05/results"
MODEL_DIR           = "/workspace/NLP/NLP05/models"

# --- 데이터 파일 ---
SFT_DATA_PATH       = DATA_DIR + "/kochatgpt_1_SFT.jsonl"
RM_DATA_PATH        = DATA_DIR + "/kochatgpt_2_RM.jsonl"
PPO_DATA_PATH       = DATA_DIR + "/kochatgpt_3_PPO.jsonl"

# --- Foundation Model ---
BASE_MODEL_NAME     = "skt/kogpt2-base-v2"

# --- SFT 하이퍼파라미터 ---
SFT_EPOCHS          = 3
SFT_BATCH_SIZE      = 8
SFT_LR              = 2e-5
SFT_MAX_LEN         = 128
SFT_WARMUP_STEPS    = 100
SFT_SAVE_PATH       = MODEL_DIR + "/sft_model"

# --- RM 하이퍼파라미터 ---
RM_EPOCHS           = 3
RM_BATCH_SIZE       = 8
RM_LR               = 1e-5
RM_MAX_LEN          = 128
RM_SAVE_PATH        = MODEL_DIR + "/rm_model"

# --- PPO 하이퍼파라미터 ---
PPO_EPOCHS          = 1
PPO_BATCH_SIZE      = 4
PPO_LR              = 1e-5
PPO_MAX_LEN         = 128
PPO_KL_COEF         = 0.1       # KL divergence 페널티 계수
PPO_CLIP_EPS        = 0.2       # TRPO 클리핑 범위 (1 ± ε)
PPO_SAVE_PATH       = MODEL_DIR + "/ppo_model"

# --- Decoding 실험 파라미터 ---
DECODING_CONFIGS = {
    "greedy":    {"do_sample": False},
    "beam_2":    {"num_beams": 2,  "do_sample": False},
    "beam_4":    {"num_beams": 4,  "do_sample": False},
    "topk_10":   {"do_sample": True, "top_k": 10},
    "topk_50":   {"do_sample": True, "top_k": 50},
    "topp_09":   {"do_sample": True, "top_p": 0.9},
}

# --- 공통 ---
RANDOM_SEED         = 42
DEVICE              = "cuda"
