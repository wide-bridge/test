"""
03_pretrain.py
mini BERT (vocab_size=8000, ~1M params) 사전학습
MLM loss + NSP loss, WarmupLinearSchedule
"""

import os
os.chdir("/workspace/NLP/NLP04/miniBERT")

import json
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.join('/workspace/NLP/NLP04/miniBERT', 'scripts'))
from experiment_tracker import ExperimentTracker

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = '/workspace/NLP/NLP04/miniBERT'
PROC_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    CFG = json.load(f)

VOCAB_SIZE  = CFG['vocab_size']
D_MODEL     = CFG['d_model']
NUM_HEADS   = CFG['num_heads']
NUM_LAYERS  = CFG['num_layers']
D_FF        = CFG['d_ff']
MAX_SEQ_LEN = CFG['max_seq_len']
DROPOUT     = CFG['dropout']
BATCH_SIZE  = CFG['batch_size']
EPOCHS      = CFG['epochs']
LR          = CFG['learning_rate']

PAD_ID = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] {device}")


# ══════════════════════════════════════════════════════════════
# 1. Utility
# ══════════════════════════════════════════════════════════════

def get_pad_mask(seq: torch.Tensor, pad_idx: int = PAD_ID) -> torch.Tensor:
    """
    seq : (B, L)
    Returns key_padding_mask : (B, L)  True where pad
    """
    return seq == pad_idx


# ══════════════════════════════════════════════════════════════
# 2. GELU Activation
# ══════════════════════════════════════════════════════════════

class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ══════════════════════════════════════════════════════════════
# 3. BERTEmbedding
# ══════════════════════════════════════════════════════════════

class BERTEmbedding(nn.Module):
    """
    Token Embedding + Positional Embedding + Segment Embedding + LayerNorm
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 n_segments: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_emb   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb     = nn.Embedding(max_seq_len, d_model)
        self.segment_emb = nn.Embedding(n_segments, d_model)
        self.layer_norm  = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout     = nn.Dropout(dropout)

        # 위치 인덱스 버퍼 (학습되는 positional embedding)
        positions = torch.arange(max_seq_len).unsqueeze(0)  # (1, L)
        self.register_buffer('positions', positions)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids   : (B, L)
        segment_ids : (B, L)
        Returns     : (B, L, D)
        """
        L = input_ids.size(1)
        pos = self.positions[:, :L]   # (1, L)

        x = self.token_emb(input_ids) + self.pos_emb(pos) + self.segment_emb(segment_ids)
        return self.dropout(self.layer_norm(x))


# ══════════════════════════════════════════════════════════════
# 4. MultiHeadAttention
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.scale     = math.sqrt(self.d_head)

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x                : (B, L, D)
        key_padding_mask : (B, L)  True where padded
        Returns          : (B, L, D)
        """
        B, L, D = x.shape
        H, DH   = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, L, H, DH).transpose(1, 2)   # (B, H, L, DH)
        K = self.k_proj(x).view(B, L, H, DH).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, DH).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if key_padding_mask is not None:
            # (B, L) → (B, 1, 1, L)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)                            # (B, H, L, DH)
        out    = out.transpose(1, 2).contiguous().view(B, L, D)   # (B, L, D)
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════
# 5. TransformerEncoderLayer
# ══════════════════════════════════════════════════════════════

class TransformerEncoderLayer(nn.Module):
    """
    Self-Attention → Add & Norm → FFN → Add & Norm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1       = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2       = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1    = nn.Dropout(dropout)
        self.dropout2    = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention + residual
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════
# 6. BERTModel
# ══════════════════════════════════════════════════════════════

class BERTModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len,
                                        dropout=dropout)
        self.encoders  = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.pool_fc   = nn.Linear(d_model, d_model)
        self.pool_act  = nn.Tanh()

    def forward(self, input_ids: torch.Tensor,
                segment_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """
        Returns:
            sequence_output : (B, L, D)
            pooled_output   : (B, D)  — [CLS] 토큰 기반
        """
        key_padding_mask = get_pad_mask(input_ids)   # (B, L)

        x = self.embedding(input_ids, segment_ids)   # (B, L, D)
        for enc in self.encoders:
            x = enc(x, key_padding_mask=key_padding_mask)

        cls_token   = x[:, 0, :]                          # (B, D)
        pooled      = self.pool_act(self.pool_fc(cls_token))
        return x, pooled


# ══════════════════════════════════════════════════════════════
# 7. BERTForPretraining
# ══════════════════════════════════════════════════════════════

class BERTForPretraining(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.bert = BERTModel(vocab_size, d_model, num_heads, num_layers,
                              d_ff, max_seq_len, dropout)

        # MLM Head: hidden → vocab logits
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Linear(d_model, vocab_size),
        )

        # NSP Head: pooled → 2-class
        self.nsp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, input_ids, segment_ids, attention_mask):
        seq_out, pooled = self.bert(input_ids, segment_ids, attention_mask)
        mlm_logits  = self.mlm_head(seq_out)    # (B, L, V)
        nsp_logits  = self.nsp_head(pooled)     # (B, 2)
        return mlm_logits, nsp_logits


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class BERTPretrainDataset(Dataset):
    def __init__(self, proc_dir: str, max_seq_len: int, n_samples: int):
        self.n        = n_samples
        self.L        = max_seq_len

        self.input_ids      = np.memmap(os.path.join(proc_dir, 'input_ids.dat'),
                                        dtype='int32', mode='r', shape=(n_samples, max_seq_len))
        self.segment_ids    = np.memmap(os.path.join(proc_dir, 'segment_ids.dat'),
                                        dtype='int32', mode='r', shape=(n_samples, max_seq_len))
        self.attention_mask = np.memmap(os.path.join(proc_dir, 'attention_mask.dat'),
                                        dtype='int32', mode='r', shape=(n_samples, max_seq_len))
        self.mlm_labels     = np.memmap(os.path.join(proc_dir, 'mlm_labels.dat'),
                                        dtype='int32', mode='r', shape=(n_samples, max_seq_len))
        self.nsp_label      = np.memmap(os.path.join(proc_dir, 'nsp_label.dat'),
                                        dtype='int32', mode='r', shape=(n_samples,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            'input_ids':      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            'segment_ids':    torch.tensor(self.segment_ids[idx],    dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'mlm_labels':     torch.tensor(self.mlm_labels[idx],     dtype=torch.long),
            'nsp_label':      torch.tensor(int(self.nsp_label[idx]), dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════
# LR Scheduler: WarmupLinearSchedule
# ══════════════════════════════════════════════════════════════

class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)
        super().__init__(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════
# 학습 / 평가
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler,
                mlm_criterion, nsp_criterion):
    model.train()
    total_loss = total_mlm = total_nsp = 0.0
    total_mlm_correct = total_mlm_tokens = 0
    total_nsp_correct = total_nsp_samples = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        segment_ids    = batch['segment_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        mlm_labels     = batch['mlm_labels'].to(device)
        nsp_label      = batch['nsp_label'].to(device)

        mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)

        # MLM loss
        mlm_loss = mlm_criterion(
            mlm_logits.view(-1, VOCAB_SIZE),
            mlm_labels.view(-1)
        )

        # NSP loss
        nsp_loss = nsp_criterion(nsp_logits, nsp_label)

        loss = mlm_loss + nsp_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm  += mlm_loss.item()
        total_nsp  += nsp_loss.item()

        # MLM accuracy (ignore_index=-100인 위치 제외)
        mlm_mask = (mlm_labels != -100)
        pred_mlm = mlm_logits.argmax(dim=-1)
        total_mlm_correct += (pred_mlm[mlm_mask] == mlm_labels[mlm_mask]).sum().item()
        total_mlm_tokens  += mlm_mask.sum().item()

        # NSP accuracy
        pred_nsp = nsp_logits.argmax(dim=-1)
        total_nsp_correct += (pred_nsp == nsp_label).sum().item()
        total_nsp_samples += nsp_label.size(0)

    n = len(loader)
    return {
        'total_loss': total_loss / n,
        'mlm_loss':   total_mlm / n,
        'nsp_loss':   total_nsp / n,
        'mlm_acc':    total_mlm_correct / max(1, total_mlm_tokens),
        'nsp_acc':    total_nsp_correct / max(1, total_nsp_samples),
    }


# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════

def save_loss_plot(history: list):
    epochs     = [h['epoch'] for h in history]
    mlm_losses = [h['mlm_loss'] for h in history]
    nsp_losses = [h['nsp_loss'] for h in history]
    tot_losses = [h['total_loss'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, title, color in zip(
        axes,
        [tot_losses, mlm_losses, nsp_losses],
        ['Total Loss', 'MLM Loss', 'NSP Loss'],
        ['steelblue', 'darkorange', 'green'],
    ):
        ax.plot(epochs, vals, marker='o', color=color)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.suptitle('mini BERT Pretraining Loss', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, 'pretrain_loss.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved → {save_path}")


# ══════════════════════════════════════════════════════════════
# 인터랙티브 제어
# ══════════════════════════════════════════════════════════════

def interactive_config_update(cfg: dict) -> dict:
    """훈련 시작 전 하이퍼파라미터를 항목별로 확인하고 변경한다."""
    EDITABLE = [
        ('d_model',       int,   'd_model'),
        ('num_heads',     int,   'num_heads'),
        ('num_layers',    int,   'num_layers'),
        ('d_ff',          int,   'd_ff'),
        ('batch_size',    int,   'batch_size'),
        ('epochs',        int,   'epochs'),
        ('learning_rate', float, 'learning_rate'),
        ('dropout',       float, 'dropout'),
    ]

    print("\n" + "=" * 60)
    print("  하이퍼파라미터 설정  (Enter = 기본값 유지)")
    print("=" * 60)

    changed = {}
    for key, cast, label in EDITABLE:
        current = cfg.get(key)
        try:
            val = input(f"  {label} [{current}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if val:
            try:
                new_val = cast(val)
                cfg[key] = new_val
                changed[key] = (current, new_val)
            except ValueError:
                print(f"  [skip] 잘못된 입력 '{val}' — 기본값 {current} 유지")

    config_path = os.path.join(BASE_DIR, 'config.json')
    if changed:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  [저장] config.json 업데이트:")
        for k, (old, new) in changed.items():
            print(f"    {k}: {old} → {new}")
    else:
        print("\n  [변경 없음] config.json 유지")

    return cfg


def ask_resume(model_dir: str):
    """
    체크포인트 존재 시 이어서 훈련 여부를 묻는다.

    Returns:
        resume      (bool)
        ckpt_path   (str | None)
        start_epoch (int)    — 1-based, 이어서 시작할 epoch
        best_loss   (float)
    """
    best_path  = os.path.join(model_dir, 'bert_best.pt')
    final_path = os.path.join(model_dir, 'bert_final.pt')

    ckpt_path = None
    if os.path.exists(best_path):
        ckpt_path = best_path
    elif os.path.exists(final_path):
        ckpt_path = final_path

    if ckpt_path is None:
        return False, None, 1, float('inf')

    print(f"\n[checkpoint] 기존 체크포인트가 감지되었습니다: {os.path.basename(ckpt_path)}")
    try:
        ans = input("  이어서 훈련하시겠습니까? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = 'n'
        print()

    if ans == 'y':
        ckpt        = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss   = ckpt.get('loss', float('inf'))
        print(f"  → epoch {start_epoch}부터 이어서 학습 (best_loss={best_loss:.4f})")
        return True, ckpt_path, start_epoch, best_loss
    else:
        print("  → 처음부터 새로 학습합니다.")
        return False, None, 1, float('inf')


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    global VOCAB_SIZE   # train_epoch 내부에서 참조

    print("=" * 60)
    print(" Step 3: mini BERT Pretraining")
    print("=" * 60)

    # ── 1. 하이퍼파라미터 인터랙티브 확인/변경 ───────────────
    cfg = interactive_config_update(CFG)

    # 업데이트된 config로 로컬 변수 재추출
    VOCAB_SIZE  = cfg['vocab_size']
    d_model     = cfg['d_model']
    num_heads   = cfg['num_heads']
    num_layers  = cfg['num_layers']
    d_ff        = cfg['d_ff']
    max_seq_len = cfg['max_seq_len']
    dropout     = cfg['dropout']
    batch_size  = cfg['batch_size']
    epochs      = cfg['epochs']
    lr          = cfg['learning_rate']

    # ── 2. 데이터 로드 ────────────────────────────────────────
    meta_path = os.path.join(PROC_DIR, 'dataset_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    n_samples = meta['n_samples']
    print(f"\n[data] {n_samples:,} samples, max_seq_len={max_seq_len}")

    dataset = BERTPretrainDataset(PROC_DIR, max_seq_len, n_samples)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # ── 3. 모델 초기화 ────────────────────────────────────────
    model = BERTForPretraining(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] Total parameters : {total_params:,}")
    print(f"[model] Trainable params  : {train_params:,}")

    # ── 4. 이어서 훈련 여부 확인 ─────────────────────────────
    resume, ckpt_path, start_epoch, best_loss = ask_resume(MODEL_DIR)
    best_epoch = start_epoch - 1   # 아직 아무 epoch도 완료 안 한 상태

    if resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"  [loaded] model weights from {os.path.basename(ckpt_path)}")

    # ── 5. ExperimentTracker 초기화 ──────────────────────────
    tracker = ExperimentTracker(cfg=cfg, total_params=total_params)

    # ── 6. Optimizer & Scheduler ──────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps  = len(loader) * epochs
    warmup_steps = int(total_steps * 0.10)

    if resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        # 이미 완료된 step만큼 scheduler 앞으로 이동
        scheduler      = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)
        resumed_steps  = (start_epoch - 1) * len(loader)
        scheduler.last_epoch = resumed_steps - 1
        scheduler.step()
        print(f"  [scheduler] resumed at step {resumed_steps}")
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)

    print(f"[scheduler] total_steps={total_steps}, warmup_steps={warmup_steps}")

    # ── 7. Loss ───────────────────────────────────────────────
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()

    # ── 8. 학습 루프 ──────────────────────────────────────────
    history    = []
    if best_loss == float('inf'):
        best_loss  = float('inf')
        best_epoch = -1

    for epoch in range(start_epoch, epochs + 1):
        t0      = time.time()
        stats   = train_epoch(model, loader, optimizer, scheduler,
                              mlm_criterion, nsp_criterion)
        elapsed = time.time() - t0

        stats['epoch'] = epoch
        history.append(stats)
        tracker.log_epoch(epoch, stats)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"total={stats['total_loss']:.4f} | "
            f"mlm={stats['mlm_loss']:.4f} | "
            f"nsp={stats['nsp_loss']:.4f} | "
            f"mlm_acc={stats['mlm_acc']:.4f} | "
            f"nsp_acc={stats['nsp_acc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # epoch 결과 json 저장
        epoch_log_path = os.path.join(MODEL_DIR, f'epoch_{epoch:02d}_log.json')
        with open(epoch_log_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # best model 저장
        if stats['total_loss'] < best_loss:
            best_loss  = stats['total_loss']
            best_epoch = epoch
            ckpt_save  = os.path.join(MODEL_DIR, 'bert_best.pt')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'loss':        best_loss,
                'config':      cfg,
            }, ckpt_save)
            print(f"  → best checkpoint saved (epoch={epoch}, loss={best_loss:.4f})")

    # ── 9. 최종 체크포인트 ────────────────────────────────────
    final_path = os.path.join(MODEL_DIR, 'bert_final.pt')
    torch.save({'epoch': epochs, 'model_state': model.state_dict(), 'config': cfg},
               final_path)

    # ── 10. 학습 히스토리 저장 ────────────────────────────────
    hist_path = os.path.join(MODEL_DIR, 'train_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # ── 11. 시각화 ────────────────────────────────────────────
    save_loss_plot(history)

    # ── 12. ExperimentTracker 저장 ────────────────────────────
    tracker.save(best_epoch=best_epoch)

    print("\n" + "=" * 60)
    print(f" Pretraining Complete!")
    print(f" Best epoch : {best_epoch}  (loss={best_loss:.4f})")
    print(f" Total params: {total_params:,}")
    print("=" * 60)


if __name__ == '__main__':
    main()
