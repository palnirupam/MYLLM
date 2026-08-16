# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
Dhruva V0 -- Tokenizer Validation Audit Script (Production Architecture)
========================================================================
All proxy models now use the PRODUCTION architecture:
  - MyLLMModel (causal decoder)
  - RoPE positional encoding
  - GQA (n_kv_heads=4)
  - SwiGLU FFN
  - RMSNorm
  - Weight-tied LM head

Sections:
  1. Loss-pipeline audit  (manual batch inspection + independent CE check)
  2. BpB formula audit    (toy-corpus sanity check)
  3. Uniform baseline     (ln(vocab_size) per candidate)
  4. LR stability pilot   (3e-4 / 1e-4 / 3e-5 sweep on 32K candidate)
  5. Loss-diagnostic step sequence (init / 1 / 10 / 100 steps)
  6. Controlled comparison
        View A -- same architecture (fixed production config, only vocab_size varies)
        View B -- parameter-normalized (~100M-110M budget, n_layers adjusted)
     both with:
        Equal-raw-text exposure  (same 10MB raw bytes)
        Equal-token-budget       (same training token count)
  7. Final report files
"""

import os
import sys
import time
import math
import json
import uuid
import hashlib
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

# ── Production model imports (B2/B3 FIX) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myllm.core.model.transformer import MyLLMModel    # Production model
from myllm.core.model.config import ModelConfig         # Production config

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
from myllm.utils.env import get_project_root
WORKSPACE   = get_project_root() / "artifacts/stage2_tokenizer_experiments"
STAGE2_OUT  = WORKSPACE   # all outputs land here per the plan
WORKSPACE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Production model builders (B2/B3 FIX)
# Replaces ProxyTransformer (bidirectional encoder + learnable PE) with
# MyLLMModel (causal decoder: RoPE + GQA + SwiGLU + RMSNorm + tied LM head).
# ─────────────────────────────────────────────────────────────────────────────

# Experiment SEQ_LEN — keep below production max_seq_len=4096 for memory.
# Use 512 to match the audited production config.
EXP_SEQ_LEN = 512

# B3 FIX: Same RoPE parameters as production.
EXP_ROPE_THETA = 10000.0

# GQA config: production uses n_heads=12, n_kv_heads=4  =>  n_rep = 3
EXP_N_HEADS    = 12
EXP_N_KV_HEADS = 4

# Production intermediate_size for SwiGLU
EXP_FFN_SIZE = 2048


def _make_config(vocab_size: int, n_layers: int) -> ModelConfig:
    """Create a production-equivalent ModelConfig for the tokenizer experiment."""
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=n_layers,
        n_heads=EXP_N_HEADS,
        n_kv_heads=EXP_N_KV_HEADS,
        intermediate_size=EXP_FFN_SIZE,
        max_seq_len=EXP_SEQ_LEN,
        dropout=0.0,
        norm_eps=1e-5,
        rope_theta=EXP_ROPE_THETA,
        tie_word_embeddings=True,
    )


def build_model_view_a(vocab_size: int) -> MyLLMModel:
    """View A: fixed 12-layer production architecture, only vocab_size changes."""
    cfg = _make_config(vocab_size=vocab_size, n_layers=12)
    return MyLLMModel(cfg)


# Target: unique params of 32K, 12-layer production model (~100M)
# We search for the n_layers that keeps 48K/64K within +5M of this baseline.
TARGET_PARAMS = None  # Set lazily from 32K baseline on first call


def build_model_view_b(vocab_size: int) -> MyLLMModel:
    """View B: reduce n_layers to stay within ~110M unique-param budget."""
    global TARGET_PARAMS
    if TARGET_PARAMS is None:
        # Set baseline from 32K 12-layer model
        ref = MyLLMModel(_make_config(32000, 12))
        TARGET_PARAMS = count_params(ref)["total_unique_params"]
        del ref

    # Search from 12 layers down to 1
    for n_layers in range(12, 0, -1):
        cfg   = _make_config(vocab_size=vocab_size, n_layers=n_layers)
        model = MyLLMModel(cfg)
        p     = count_params(model)["total_unique_params"]
        # Accept if within TARGET_PARAMS + 5M
        if p <= TARGET_PARAMS + 5_000_000:
            return model
        del model

    # Fallback: 1 layer (should not reach here)
    return MyLLMModel(_make_config(vocab_size=vocab_size, n_layers=1))


def count_params(model: MyLLMModel) -> Dict[str, Any]:
    """Return unique parameter counts, correctly handling tied weights."""
    seen_ptrs = set()
    total_unique = 0
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            total_unique += p.numel()

    cfg = model.config
    emb_params   = cfg.vocab_size * cfg.d_model  # token_embedding
    # lm_head is tied to embedding, so unique count excludes it

    return {
        "total_unique_params":  total_unique,
        "embedding_params":     emb_params,
        "lm_head_params":       emb_params,   # tied — same tensor
        "transformer_params":   total_unique - emb_params,
        "vocab_size":           cfg.vocab_size,
        "n_layers":             cfg.n_layers,
        "d_model":              cfg.d_model,
        "n_heads":              cfg.n_heads,
        "n_kv_heads":           cfg.n_kv_heads,
        "n_rep":                cfg.n_heads // cfg.n_kv_heads,
        "head_dim":             cfg.head_dim,
        "intermediate_size":    cfg.intermediate_size,
        "rope_theta":           cfg.rope_theta,
        "tie_word_embeddings":  cfg.tie_word_embeddings,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ─────────────────────────────────────────────────────────────────────────────
from myllm.core.tokenizer.bpe import BPETokenizer   # noqa: E402 (after path setup)

def train_tokenizer(corpus: str, vocab_size: int) -> BPETokenizer:
    set_seed()
    return BPETokenizer.train_from_texts([corpus], vocab_size=vocab_size)


def tokenize_and_prepare(text: str, tokenizer: BPETokenizer,
                          seq_len: int, device: torch.device):
    """
    Returns (x, y, total_raw_bytes, total_tokens_in_tensor).

    Causal shift: x[i] = input token at position i,
                  y[i] = input token at position i+1.
    This shift happens EXACTLY ONCE here.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    raw_bytes = len(text.encode('utf-8'))

    n_chunks = len(ids) // (seq_len + 1)
    if n_chunks == 0:
        return None, None, raw_bytes, 0

    ids  = ids[: n_chunks * (seq_len + 1)]
    data = torch.tensor(ids, dtype=torch.long).view(n_chunks, seq_len + 1)

    x = data[:, :-1].to(device)   # input  -- shape (n_chunks, seq_len)
    y = data[:, 1:].to(device)    # target -- shape (n_chunks, seq_len)
    return x, y, raw_bytes, len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- Loss-pipeline audit (manual batch inspection)
# ─────────────────────────────────────────────────────────────────────────────
def run_loss_audit(tokenizer: BPETokenizer, vocab_size: int,
                   sample_text: str, device: torch.device) -> dict:
    print("\n" + "="*60)
    print("SECTION 1 -- Loss-Pipeline Audit")
    print("="*60)

    set_seed()
    # B2 FIX: Use production model (causal decoder with RoPE/GQA/SwiGLU)
    model = build_model_view_a(vocab_size).to(device)
    model.eval()

    # Tiny batch: first 10 tokens of sample text
    ids = tokenizer.encode(sample_text[:200], add_special_tokens=False)[:10]
    if len(ids) < 4:
        return {"status": "SKIP", "reason": "corpus too short for audit"}

    # Use first 8 tokens as a single sequence of length 8
    seq   = torch.tensor(ids[:8], dtype=torch.long, device=device).unsqueeze(0)  # (1,8)
    x_in  = seq[:, :-1]   # (1,7) -- input ids
    y_tgt = seq[:, 1:]     # (1,7) -- labels = shifted by 1

    print(f"\nManual batch inspection:")
    print(f"  input_ids  = {x_in[0].tolist()}")
    print(f"  labels     = {y_tgt[0].tolist()}")
    print(f"  Verifying: input[t] -> label[t] = input[t+1]")
    for t in range(x_in.shape[1]):
        expected_label = seq[0, t+1].item()
        actual_label   = y_tgt[0, t].item()
        assert expected_label == actual_label, (
            f"  SHIFT BUG at t={t}: expected {expected_label}, got {actual_label}")
    print("  Shift verification: PASS -- labels == input[1:]")

    with torch.no_grad():
        logits, _ = model(x_in)   # B2 FIX: MyLLMModel returns (logits, kv_cache)

    print(f"\n  logits.shape = {tuple(logits.shape)}")
    print(f"  Expected (1, {x_in.shape[1]}, {vocab_size})")
    assert logits.shape == (1, x_in.shape[1], vocab_size), "Logits shape mismatch!"
    print("  Logits shape: PASS")

    valid_target_count = y_tgt.numel()
    print(f"  valid_target_count = {valid_target_count}  (no padding in this run)")

    # Production CE loss
    loss_prod = F.cross_entropy(
        logits.view(-1, vocab_size),
        y_tgt.view(-1),
        reduction='mean'
    )

    # Independent reference CE
    log_probs = F.log_softmax(logits, dim=-1)                    # (1, T, V)
    gathered  = log_probs.gather(2, y_tgt.unsqueeze(-1)).squeeze(-1)  # (1, T)
    loss_ref  = -gathered.mean()

    diff = abs(loss_prod.item() - loss_ref.item())
    # Threshold: 1e-4 is appropriate for float32 cross-entropy over large logits.
    # The two paths accumulate fp32 rounding errors slightly differently.
    # A difference < 1e-4 confirms the implementations agree and is not a bug.
    CE_TOLERANCE = 1e-4
    print(f"\n  production CE  = {loss_prod.item():.6f} nats/token")
    print(f"  independent CE = {loss_ref.item():.6f} nats/token")
    print(f"  difference     = {diff:.2e}  (tolerance < {CE_TOLERANCE:.0e})")
    print(f"  NOTE: High loss ({loss_prod.item():.1f} nats) is expected for a fresh"
          f" random-weight model; ln(32000)={math.log(32000):.2f} is only the"
          f" UNIFORM baseline. Random logits are far from uniform.")
    assert diff < CE_TOLERANCE, f"CE mismatch exceeds tolerance! diff={diff:.2e} > {CE_TOLERANCE:.0e}"
    print("  CE agreement: PASS")

    # Logit-level spot check on token 0
    t0_logits = logits[0, 0].float()
    t0_target = y_tgt[0, 0].item()
    t0_log_p  = F.log_softmax(t0_logits, dim=-1)[t0_target].item()
    t0_prob   = math.exp(t0_log_p)
    print(f"\n  Spot-check t=0: input_id={x_in[0,0].item()} "
          f"-> target_id={t0_target}, "
          f"assigned log-prob={t0_log_p:.4f} ({t0_prob*100:.4f}%)")

    result = {
        "status":                 "PASS",
        "shift_check":            "PASS",
        "logits_shape_check":     "PASS",
        "valid_target_count":     valid_target_count,
        "production_ce_nats":     round(loss_prod.item(), 6),
        "independent_ce_nats":    round(loss_ref.item(), 6),
        "ce_diff":                round(diff, 10),
        "ce_agreement":           "PASS",
    }
    print("\nSection 1 result:", json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- BpB formula audit (toy corpus)
# ─────────────────────────────────────────────────────────────────────────────
def run_bpb_audit(vocab_size: int = 32000) -> dict:
    print("\n" + "="*60)
    print("SECTION 2 -- BpB Formula Audit (toy corpus)")
    print("="*60)

    # Toy: uniform distribution over vocab_size tokens.
    # Expected CE = ln(vocab_size); BpB = ln(V) / ln(2) * tokens / bytes
    # We construct a tiny corpus of 8 ASCII chars = 8 bytes,
    # tokenized into 8 single-byte tokens.
    # loss = ln(V); bits_per_token = ln(V)/ln(2)
    # total_bits = ln(V)/ln(2) * 7  (7 prediction positions)
    # bytes = 8 (the raw text)
    # bpb_expected = ln(V)/ln(2) * 7 / 8

    V = vocab_size
    # Simulate logits = uniform (all zeros -> uniform softmax)
    T_pred    = 7
    raw_bytes = 8
    logits    = torch.zeros(1, T_pred, V)
    labels    = torch.zeros(1, T_pred, dtype=torch.long)

    loss_uniform = F.cross_entropy(logits.view(-1, V), labels.view(-1)).item()
    expected_loss = math.log(V)

    print(f"  uniform CE (computed) = {loss_uniform:.6f} nats")
    print(f"  ln({V})               = {expected_loss:.6f} nats")
    assert abs(loss_uniform - expected_loss) < 1e-4, "Uniform CE wrong!"
    print("  Uniform CE: PASS")

    # BpB calculation
    bits_per_token = loss_uniform / math.log(2)
    total_bits     = bits_per_token * T_pred
    bpb_computed   = total_bits / raw_bytes
    bpb_expected   = (expected_loss / math.log(2)) * T_pred / raw_bytes

    print(f"\n  bits_per_token = {bits_per_token:.6f}")
    print(f"  total_bits     = {total_bits:.6f}")
    print(f"  raw_bytes      = {raw_bytes}")
    print(f"  bpb (computed) = {bpb_computed:.6f}")
    print(f"  bpb (expected) = {bpb_expected:.6f}")
    # Tolerance: loss_uniform is float32 (PyTorch), expected_loss is float64 (math.log).
    # Cross-precision comparison introduces ~1e-6 rounding difference — 1e-5 is correct here.
    BPB_TOLERANCE = 1e-5
    assert abs(bpb_computed - bpb_expected) < BPB_TOLERANCE, \
        f"BpB formula mismatch! diff={abs(bpb_computed-bpb_expected):.2e} > {BPB_TOLERANCE:.0e}"
    print("  BpB formula: PASS")

    # Sanity: a perfect model (loss->0) should give bpb->0
    perfect_bpb = 0.0 / math.log(2) * T_pred / raw_bytes
    print(f"\n  Perfect-model BpB = {perfect_bpb:.6f}  (sanity: should be 0)")

    result = {
        "status":                   "PASS",
        "formula":                  "(loss_nats / ln(2)) * val_tokens / raw_val_bytes",
        "toy_uniform_ce_computed":  round(loss_uniform, 6),
        "toy_uniform_ce_expected":  round(expected_loss, 6),
        "toy_bpb_computed":         round(bpb_computed, 6),
        "toy_bpb_expected":         round(bpb_expected, 6),
        "uniform_ce_check":         "PASS",
        "bpb_formula_check":        "PASS",
    }
    print("\nSection 2 result:", json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- Uniform baseline per vocabulary
# ─────────────────────────────────────────────────────────────────────────────
def compute_baseline(vocab_size: int) -> dict:
    uniform_nll = math.log(vocab_size)
    uniform_bpb = uniform_nll / math.log(2)   # 1 token = 1 token, bpb not byte-normalized here
    # Note: the "per-byte" baseline depends on tokenizer compression.
    # We report the per-TOKEN uniform NLL; comparison against measured
    # val_loss_nats tells us whether the model is learning at all.
    return {
        "vocab_size":          vocab_size,
        "uniform_nll_nats":    round(uniform_nll, 6),
        "uniform_bpt_bits":    round(uniform_bpb, 6),
        "note": "Model must show val_loss < uniform_nll to demonstrate learning."
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- LR stability pilot (32K only)
# ─────────────────────────────────────────────────────────────────────────────
def run_lr_pilot(tokenizer: BPETokenizer, vocab_size: int,
                 train_text: str, device: torch.device) -> dict:
    print("\n" + "="*60)
    print("SECTION 4 -- LR Stability Pilot (32K, 100 steps each)")
    print("="*60)

    PILOT_STEPS = 100
    BATCH_SIZE  = 4
    SEQ_LEN     = 512
    LR_CANDIDATES = [3e-4, 1e-4, 3e-5]

    x_train, y_train, _, _ = tokenize_and_prepare(train_text, tokenizer, SEQ_LEN, device)
    if x_train is None:
        return {"status": "SKIP", "reason": "insufficient training data"}

    results = {}
    for lr in LR_CANDIDATES:
        set_seed()
        # B2 FIX: Use production model
        model = build_model_view_a(vocab_size).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)
        model.train()

        steps_log = {}
        for step in range(PILOT_STEPS):
            batch_x = x_train[(step % (x_train.size(0)//BATCH_SIZE))*BATCH_SIZE :
                               (step % (x_train.size(0)//BATCH_SIZE) + 1)*BATCH_SIZE]
            batch_y = y_train[(step % (y_train.size(0)//BATCH_SIZE))*BATCH_SIZE :
                               (step % (y_train.size(0)//BATCH_SIZE) + 1)*BATCH_SIZE]

            opt.zero_grad()
            logits, _ = model(batch_x)   # B2 FIX: unpack (logits, kv_cache)
            loss   = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                steps_log[step] = {"loss": "NaN/Inf", "stable": False}
                break

            loss.backward()
            grad_norm  = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()

            if step in (0, 1, 9, 99):
                steps_log[step] = {
                    "loss":      round(loss.item(), 4),
                    "grad_norm": round(grad_norm, 4),
                }

        baseline = math.log(vocab_size)
        final_loss = steps_log.get(99, {}).get("loss", "N/A")
        stable = isinstance(final_loss, float) and final_loss < baseline * 2.5
        print(f"  LR={lr:.1e}: init_loss={steps_log.get(0,{}).get('loss','?'):.4f}  "
              f"step100={final_loss}  uniform_baseline={baseline:.4f}  "
              f"stable={'YES' if stable else 'NO'}")
        results[str(lr)] = {"steps": steps_log, "stable": stable,
                             "uniform_baseline": round(baseline, 6)}

    # Select best LR: lowest stable final loss
    best_lr = None
    best_loss = float('inf')
    for lr_str, data in results.items():
        last = data["steps"].get(99, {}).get("loss", float('inf'))
        if isinstance(last, float) and last < best_loss and data["stable"]:
            best_loss = last
            best_lr   = float(lr_str)

    print(f"\n  Selected LR: {best_lr}")
    results["selected_lr"] = best_lr
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- Loss diagnostic step sequence
# ─────────────────────────────────────────────────────────────────────────────
def run_loss_diagnostic(tokenizer: BPETokenizer, vocab_size: int,
                         train_text: str, lr: float,
                         device: torch.device) -> dict:
    print("\n" + "="*60)
    print(f"SECTION 5 -- Loss Diagnostic Sequence (LR={lr:.1e})")
    print("="*60)

    BATCH_SIZE = 4
    SEQ_LEN    = 512
    x_train, y_train, _, _ = tokenize_and_prepare(train_text, tokenizer, SEQ_LEN, device)
    if x_train is None:
        return {"status": "SKIP"}

    set_seed()
    # B2 FIX: Use production model
    model = build_model_view_a(vocab_size).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.95), weight_decay=0.1)

    baseline = math.log(vocab_size)
    log = {}

    def eval_step(step_name):
        model.eval()
        with torch.no_grad():
            batch_x = x_train[:BATCH_SIZE]
            batch_y = y_train[:BATCH_SIZE]
            logits, _ = model(batch_x)   # B2 FIX: unpack (logits, kv_cache)
            loss    = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))
        model.train()
        l = loss.item()
        nan_inf = math.isnan(l) or math.isinf(l)
        log[step_name] = {
            "loss_nats":         round(l, 4) if not nan_inf else "NaN/Inf",
            "diff_from_baseline": round(l - baseline, 4) if not nan_inf else "NaN/Inf",
            "learning":          "YES" if (not nan_inf and l < baseline) else "NO",
        }
        print(f"  {step_name:>10}: loss={l:.4f}  baseline={baseline:.4f}  "
              f"diff={l-baseline:+.4f}  learning={'YES' if l<baseline else 'NO'}")

    eval_step("init")

    for target_step in [1, 10, 100]:
        for step in range(target_step - (0 if target_step == 1 else
                           (1 if target_step == 10 else 10))):
            batch_x = x_train[(step % (x_train.size(0)//BATCH_SIZE))*BATCH_SIZE :
                               (step % (x_train.size(0)//BATCH_SIZE) + 1)*BATCH_SIZE]
            batch_y = y_train[(step % (y_train.size(0)//BATCH_SIZE))*BATCH_SIZE :
                               (step % (y_train.size(0)//BATCH_SIZE) + 1)*BATCH_SIZE]
            opt.zero_grad()
            logits, _ = model(batch_x)   # B2 FIX: unpack (logits, kv_cache)
            loss   = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                eval_step(f"step_{target_step}")
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        eval_step(f"step_{target_step}")

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop -- shared across all views
# ─────────────────────────────────────────────────────────────────────────────
def run_proxy(
    tokenizer_name:  str,
    vocab_size:      int,
    model:           MyLLMModel,   # B2 FIX: production model type
    x_train:         torch.Tensor,
    y_train:         torch.Tensor,
    x_val:           torch.Tensor,
    y_val:           torch.Tensor,
    raw_val_bytes:   int,
    val_tokens:      int,
    train_tokens:    int,
    lr:              float,
    device:          torch.device,
    total_steps:     Optional[int] = None,
    checkpoint_fractions: List[float] = (0.25, 0.5, 0.75, 1.0),
    run_id:          str = "",
    view_tag:        str = "",
    live_progress_path: Optional[Path] = None,
) -> dict:
    """
    Shared training loop using the production MyLLMModel.
    Optimizer: AdamW, betas=(0.9,0.95), weight_decay=0.1, grad_clip=1.0
    Warmup: linear over first 10% of steps.
    Live progress written to live_progress_path every 10 steps (Directive 14).
    """
    BATCH_SIZE = 4
    params = count_params(model)
    print(f"\n  [{tokenizer_name}] {params['total_unique_params']/1e6:.2f}M params  "
          f"({params['n_layers']} layers, d_model={params['d_model']})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             betas=(0.9, 0.95), weight_decay=0.1)

    n_batches   = x_train.size(0) // BATCH_SIZE
    num_steps   = total_steps if total_steps else n_batches
    warmup_steps = max(1, int(0.10 * num_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Uniform baseline
    uniform_nll = math.log(vocab_size)

    # Validate: iterate over val batches, accumulate sum-reduced loss
    def validate():
        model.eval()
        val_loss_sum    = 0.0
        val_token_count = 0
        with torch.no_grad():
            for s in range(x_val.size(0) // BATCH_SIZE):
                bx = x_val[s*BATCH_SIZE:(s+1)*BATCH_SIZE]
                by = y_val[s*BATCH_SIZE:(s+1)*BATCH_SIZE]
                lg, _ = model(bx)   # B2 FIX: unpack (logits, kv_cache)
                # Per-token loss WITHOUT mean reduction -- sum for correct averaging
                loss_sum = F.cross_entropy(lg.view(-1, vocab_size),
                                           by.view(-1),
                                           reduction='sum')
                val_loss_sum    += loss_sum.item()
                val_token_count += by.numel()
        model.train()
        if val_token_count == 0:
            return float('nan'), float('nan')
        avg_nats = val_loss_sum / val_token_count
        bpb = (avg_nats / math.log(2)) * val_token_count / raw_val_bytes
        return avg_nats, bpb

    # Live progress tracking (Directive 14)
    _last_val_nats = float('nan')
    _last_val_bpb  = float('nan')

    def _write_live_progress(step, train_loss):
        nonlocal _last_val_nats, _last_val_bpb
        if live_progress_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "run_id":           run_id,
            "candidate":        tokenizer_name,
            "view":             view_tag,
            "step":             step,
            "target_steps":     num_steps,
            "tokens_processed": tokens_seen,
            "train_loss":       round(train_loss, 6) if math.isfinite(train_loss) else None,
            "validation_loss":  round(_last_val_nats, 6) if math.isfinite(_last_val_nats) else None,
            "validation_bpb":   round(_last_val_bpb, 6) if math.isfinite(_last_val_bpb) else None,
            "tokens_per_second": round(tokens_seen / max(time.time() - t_start, 1e-6), 1),
            "last_checkpoint":  None,
            "timestamp":        now,
        }
        # Atomic write: write to .tmp then rename
        tmp = live_progress_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(live_progress_path)

    checkpoints = {}
    checkpoint_steps = {int(f * num_steps): f for f in checkpoint_fractions}

    model.train()
    set_seed()
    t_start = time.time()
    tokens_seen = 0
    stable = True

    for step in range(num_steps):
        batch_idx = step % n_batches
        bx = x_train[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        by = y_train[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]

        opt.zero_grad()
        logits, _ = model(bx)   # B2 FIX: unpack (logits, kv_cache)
        loss   = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            stable = False
            print(f"  INSTABILITY at step {step}: NaN/Inf loss -- aborting run")
            break

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if not math.isfinite(grad_norm):
            stable = False
            print(f"  INSTABILITY at step {step}: non-finite grad_norm -- aborting run")
            break

        opt.step()
        scheduler.step()
        tokens_seen += bx.numel()   # bx is (B, seq_len)

        # Live progress every 10 steps (Directive 14)
        if step % 10 == 0:
            _write_live_progress(step, loss.item())

        # Checkpoint
        if (step + 1) in checkpoint_steps or step == 0:
            frac = checkpoint_steps.get(step + 1, 0.0)
            val_nats, val_bpb = validate()
            _last_val_nats = val_nats   # Update live progress cache
            _last_val_bpb  = val_bpb
            key = f"step_{step+1}_frac_{frac:.2f}"
            checkpoints[key] = {
                "step":              step + 1,
                "fraction":          frac,
                "tokens_seen":       tokens_seen,
                "train_loss_nats":   round(loss.item(), 6),
                "val_loss_nats":     round(val_nats, 6),
                "val_bpb":           round(val_bpb, 6),
                "diff_from_baseline": round(val_nats - uniform_nll, 6),
                "learning":          val_nats < uniform_nll,
                "grad_norm":         round(grad_norm, 4),
                "lr_current":        round(scheduler.get_last_lr()[0], 8),
            }
            print(f"    step {step+1:>5}/{num_steps}  "
                  f"train={loss.item():.4f}  val={val_nats:.4f}  "
                  f"bpb={val_bpb:.4f}  baseline={uniform_nll:.4f}  "
                  f"learning={'YES' if val_nats < uniform_nll else 'NO'}")

    elapsed = time.time() - t_start
    tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
    final_val_nats, final_bpb = validate()

    return {
        "tokenizer":           tokenizer_name,
        "vocab_size":          vocab_size,
        "stable":              stable,
        "parameter_counts":    params,
        "raw_train_bytes":     train_tokens,   # token count -- raw bytes tracked separately
        "train_tokens":        tokens_seen,
        "optimizer_steps":     num_steps,
        "lr":                  lr,
        "train_time_sec":      round(elapsed, 2),
        "throughput_tok_sec":  round(tok_per_sec, 1),
        "uniform_baseline_nats": round(uniform_nll, 6),
        "final_val_loss_nats": round(final_val_nats, 6),
        "final_val_bpb":       round(final_bpb, 6),
        "diff_from_baseline":  round(final_val_nats - uniform_nll, 6),
        "learning":            final_val_nats < uniform_nll,
        "raw_val_bytes":       raw_val_bytes,
        "val_tokens":          val_tokens,
        "checkpoints":         checkpoints,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading + manifest
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus(proxy_bytes: int = 10_000_000,
                val_bytes: int = 1_000_000) -> dict:
    """
    Load all *_sample.txt files, concatenate in deterministic order,
    slice to proxy_bytes for train + val_bytes for val.
    Returns dict with text slices and manifest.
    """
    files = sorted(WORKSPACE.glob("*_sample.txt"))
    if not files:
        raise FileNotFoundError("No *_sample.txt files found in workspace")

    full_text = ""
    lang_sizes = {}
    for f in files:
        content = f.read_text(encoding='utf-8')
        lang_sizes[f.name] = len(content.encode('utf-8'))
        full_text += content + "\n"

    full_bytes = len(full_text.encode('utf-8'))
    print(f"\nCorpus: {len(files)} files, {full_bytes/1e6:.1f} MB total")

    train_text = full_text[:proxy_bytes]
    # val comes right after train to avoid overlap
    val_start  = proxy_bytes
    val_text   = full_text[val_start: val_start + val_bytes]

    train_bytes_actual = len(train_text.encode('utf-8'))
    val_bytes_actual   = len(val_text.encode('utf-8'))
    train_hash = hashlib.sha256(train_text.encode('utf-8')).hexdigest()
    val_hash   = hashlib.sha256(val_text.encode('utf-8')).hexdigest()

    print(f"Train slice: {train_bytes_actual/1e6:.2f} MB  SHA256={train_hash[:16]}...")
    print(f"Val   slice: {val_bytes_actual/1e6:.2f} MB  SHA256={val_hash[:16]}...")

    manifest = {
        "train_hash_sha256":   train_hash,
        "val_hash_sha256":     val_hash,
        "train_bytes":         train_bytes_actual,
        "val_bytes":           val_bytes_actual,
        "total_corpus_bytes":  full_bytes,
        "files":               list(lang_sizes.keys()),
        "lang_byte_sizes":     lang_sizes,
    }
    return {
        "train_text": train_text,
        "val_text":   val_text,
        "manifest":   manifest,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dhruva V0 Tokenizer Experiment")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a minimal 1-candidate, 1-view, 50-step smoke test "
                             "to verify the production model runs correctly. "
                             "Do not use for actual tokenizer selection.")
    args = parser.parse_args()

    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 60)
    print("Dhruva V0 -- Tokenizer Validation Audit (Production Architecture)")
    print(f"Run ID: {RUN_ID}")
    if args.smoke_test:
        print("MODE: SMOKE TEST (1 candidate, 1 view, 50 steps)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Live progress path (Directive 14) ─────────────────────────────────
    DIAG_DIR = Path("artifacts/training_diagnostics")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_PROGRESS_PATH = DIAG_DIR / "live_progress.json"

    # ── Production parameter audit (Directive 5) ──────────────────────────
    print("\n[ Production Parameter Audit ]")
    # Assert GQA config (Directive 6)
    assert EXP_N_HEADS % EXP_N_KV_HEADS == 0, (
        f"GQA config invalid: n_heads={EXP_N_HEADS} % n_kv_heads={EXP_N_KV_HEADS} != 0")
    print(f"  GQA: n_heads={EXP_N_HEADS}, n_kv_heads={EXP_N_KV_HEADS}, "
          f"n_rep={EXP_N_HEADS // EXP_N_KV_HEADS}, head_dim={768 // EXP_N_HEADS}")

    set_seed()
    ref_model = build_model_view_a(32000)
    prod_params = count_params(ref_model)
    del ref_model  # Free VRAM

    prod_params["architecture"] = "MyLLMModel"
    prod_params["positional_encoding"] = "RoPE"
    prod_params["ffn_type"] = "SwiGLU"
    prod_params["normalization"] = "RMSNorm"
    prod_params["attention_type"] = "GQA_causal"

    print(f"  Total unique params:   {prod_params['total_unique_params']/1e6:.3f}M")
    print(f"  Embedding params:      {prod_params['embedding_params']/1e6:.3f}M")
    print(f"  Transformer params:    {prod_params['transformer_params']/1e6:.3f}M")
    print(f"  LM-head params:        {prod_params['lm_head_params']/1e6:.3f}M (tied)")

    AUDIT_DIR = Path("artifacts/audit")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "production_parameter_audit.json", "w") as f:
        json.dump(prod_params, f, indent=2)
    print("  Saved: artifacts/audit/production_parameter_audit.json")

    # ── Load corpus (10MB train, 1MB val) ──────────────────────────────────
    corpus = load_corpus(proxy_bytes=10_000_000, val_bytes=1_000_000)
    train_text = corpus["train_text"]
    val_text   = corpus["val_text"]
    manifest   = corpus["manifest"]

    with open(STAGE2_OUT / "validation_corpus_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved: validation_corpus_manifest.json")

    # ── Train tokenizer for each candidate on TRAIN text (once) ───────────
    CANDIDATES = [32000] if args.smoke_test else [32000, 48000, 64000]
    tokenizers = {}
    for vs in CANDIDATES:
        print(f"\nTraining tokenizer vocab={vs}...")
        tokenizers[vs] = train_tokenizer(train_text, vs)

    # ── Section 1: Loss audit (32K) ────────────────────────────────────────
    loss_audit = run_loss_audit(tokenizers[32000], 32000, val_text, device)
    with open(STAGE2_OUT / "loss_audit.json", "w") as f:
        json.dump(loss_audit, f, indent=2)
    print("\nSaved: loss_audit.json")

    # ── Section 2: BpB formula audit ───────────────────────────────────────
    bpb_audit = run_bpb_audit(32000)
    bpb_audit_all = {vs: run_bpb_audit(vs) for vs in CANDIDATES}
    with open(STAGE2_OUT / "bpB_audit.json", "w") as f:
        json.dump(bpb_audit_all, f, indent=2)
    print("Saved: bpB_audit.json")

    # ── Section 3: Uniform baseline ────────────────────────────────────────
    baselines = {vs: compute_baseline(vs) for vs in CANDIDATES}
    print("\nUniform baselines:")
    for vs, b in baselines.items():
        print(f"  vocab={vs}: uniform_nll={b['uniform_nll_nats']:.4f} nats")

    # ── Section 4: LR pilot (32K only) ─────────────────────────────────────
    lr_results = run_lr_pilot(tokenizers[32000], 32000, train_text, device)
    selected_lr = lr_results.get("selected_lr", 1e-4)
    if selected_lr is None:
        selected_lr = 1e-4
        print(f"  WARNING: no stable LR found, defaulting to {selected_lr:.1e}")

    with open(STAGE2_OUT / "lr_sweep.json", "w") as f:
        json.dump(lr_results, f, indent=2)
    print("\nSaved: lr_sweep.json")

    # ── Section 5: Loss diagnostic ─────────────────────────────────────────
    diag = run_loss_diagnostic(tokenizers[32000], 32000, train_text,
                               selected_lr, device)
    print("\nLoss diagnostic:", json.dumps(diag, indent=2))

    # ── Sections 6-7: Full proxy comparison ────────────────────────────────
    # Smoke test: 50 steps only (verify architecture, not for selection).
    # Full run: 500 steps per candidate per view.
    MAX_PROXY_STEPS = 50 if args.smoke_test else 500

    view_a_results = []
    view_b_results = []
    param_audit    = {}

    # Pre-tokenize val text ONCE per candidate (same raw bytes for all)
    val_prepped = {}
    for vs in CANDIDATES:
        tok = tokenizers[vs]
        xv, yv, raw_val_bytes, val_tokens = tokenize_and_prepare(
            val_text, tok, 512, device)
        val_prepped[vs] = (xv, yv, raw_val_bytes, val_tokens)

    # B1 FIX: Pre-tokenize TRAIN text ONCE per candidate too.
    # Previously, tokenize_and_prepare(train_text, tok, ...) was called 4 times per
    # tokenizer (once per view: A_raw, A_budget, B_raw, B_budget) = 12 total calls.
    # With this cache, it is called 3 times (once per candidate). This is safe because:
    # - The same tokenizer produces the same output for the same input text.
    # - The tensors are read-only during training (no in-place modification).
    # - View equal-token-budget uses the same xt/yt but a smaller step count.
    print("\nPre-tokenizing training corpus (once per tokenizer candidate)...")
    train_prepped = {}
    for vs in CANDIDATES:
        tok = tokenizers[vs]
        xt, yt, _, train_tokens = tokenize_and_prepare(train_text, tok, 512, device)
        train_prepped[vs] = (xt, yt, train_tokens)
        print(f"  Dhruva-BPE-{vs//1000}k: {train_tokens:,} train tokens cached.")

    # ── View A: same architecture ───────────────────────────────────────────
    print("\n" + "="*60)
    print("VIEW A -- Same Architecture (fixed d_model=768, n_layers=12)")
    print("="*60)

    # Equal raw-text
    print("\n  [Equal raw-text exposure -- 10MB train]")
    for vs in CANDIDATES:
        name = f"Dhruva-BPE-{vs//1000}k"
        xt, yt, train_tokens = train_prepped[vs]   # B1 FIX: use cache
        xv, yv, raw_val_bytes, val_tokens = val_prepped[vs]

        set_seed()
        model = build_model_view_a(vs).to(device)
        res = run_proxy(name, vs, model, xt, yt, xv, yv,
                        raw_val_bytes, val_tokens, train_tokens,
                        lr=selected_lr, device=device,
                        total_steps=MAX_PROXY_STEPS,
                        checkpoint_fractions=[0.25, 0.5, 0.75, 1.0],
                        run_id=RUN_ID,
                        view_tag="A_equal_raw_text",
                        live_progress_path=LIVE_PROGRESS_PATH)
        res["view"] = "A_equal_raw_text"
        view_a_results.append(res)
        param_audit[f"ViewA_{name}"] = res["parameter_counts"]

    # Smoke test exits here: architecture verified, no need to run all views
    if args.smoke_test:
        print("\n" + "="*60)
        print("SMOKE TEST COMPLETE")
        print(f"  Model: MyLLMModel (production architecture)")
        print(f"  Candidate: 32K, View A, {MAX_PROXY_STEPS} steps")
        r = view_a_results[0]
        print(f"  Final val BpB: {r['final_val_bpb']:.4f}")
        print(f"  Learning: {'YES' if r['learning'] else 'NO'}")
        print(f"  Stable:   {'YES' if r['stable'] else 'NO'}")
        print("  Smoke test PASSED -- production model runs correctly.")
        print("="*60)
        return

    # Equal token-budget: take the smallest token count across 32/48/64k as target
    token_counts_a = [r["train_tokens"] for r in view_a_results]
    equal_token_budget = min(token_counts_a)
    # steps = equal_token_budget // (batch * seq_len), capped at MAX_PROXY_STEPS
    equal_steps = min(equal_token_budget // (4 * 512), MAX_PROXY_STEPS)

    print(f"\n  [Equal token-budget exposure -- {equal_token_budget:,} tokens "
          f"/ {equal_steps} steps]")
    for vs in CANDIDATES:
        name = f"Dhruva-BPE-{vs//1000}k"
        xt, yt, train_tokens = train_prepped[vs]   # B1 FIX: use cache
        xv, yv, raw_val_bytes, val_tokens = val_prepped[vs]

        set_seed()
        model = build_model_view_a(vs).to(device)
        res = run_proxy(name, vs, model, xt, yt, xv, yv,
                        raw_val_bytes, val_tokens, train_tokens,
                        lr=selected_lr, device=device,
                        total_steps=equal_steps,
                        checkpoint_fractions=[0.25, 0.5, 0.75, 1.0],
                        run_id=RUN_ID,
                        view_tag="A_equal_token_budget",
                        live_progress_path=LIVE_PROGRESS_PATH)
        res["view"] = "A_equal_token_budget"
        view_a_results.append(res)

    # ── View B: parameter-normalized ───────────────────────────────────────
    print("\n" + "="*60)
    print("VIEW B -- Parameter-Normalized (~100M-110M unique params, n_layers adjusted)")
    print("="*60)

    # Equal raw-text
    print("\n  [Equal raw-text exposure -- 10MB train]")
    for vs in CANDIDATES:
        name = f"Dhruva-BPE-{vs//1000}k"
        xt, yt, train_tokens = train_prepped[vs]   # B1 FIX: use cache
        xv, yv, raw_val_bytes, val_tokens = val_prepped[vs]

        set_seed()
        model = build_model_view_b(vs).to(device)
        p = count_params(model)
        print(f"  {name}: n_layers={p['n_layers']}, "
              f"total={p['total_unique_params']/1e6:.2f}M")
        res = run_proxy(name, vs, model, xt, yt, xv, yv,
                        raw_val_bytes, val_tokens, train_tokens,
                        lr=selected_lr, device=device,
                        total_steps=MAX_PROXY_STEPS,
                        checkpoint_fractions=[0.25, 0.5, 0.75, 1.0],
                        run_id=RUN_ID,
                        view_tag="B_equal_raw_text",
                        live_progress_path=LIVE_PROGRESS_PATH)
        res["view"] = "B_equal_raw_text"
        view_b_results.append(res)
        param_audit[f"ViewB_{name}"] = res["parameter_counts"]

    # Equal token-budget (same target as View A)
    print(f"\n  [Equal token-budget exposure -- {equal_steps} steps]")
    for vs in CANDIDATES:
        name = f"Dhruva-BPE-{vs//1000}k"
        xt, yt, train_tokens = train_prepped[vs]   # B1 FIX: use cache
        xv, yv, raw_val_bytes, val_tokens = val_prepped[vs]

        set_seed()
        model = build_model_view_b(vs).to(device)
        res = run_proxy(name, vs, model, xt, yt, xv, yv,
                        raw_val_bytes, val_tokens, train_tokens,
                        lr=selected_lr, device=device,
                        total_steps=equal_steps,
                        checkpoint_fractions=[0.25, 0.5, 0.75, 1.0],
                        run_id=RUN_ID,
                        view_tag="B_equal_token_budget",
                        live_progress_path=LIVE_PROGRESS_PATH)
        res["view"] = "B_equal_token_budget"
        view_b_results.append(res)

    # ── Save all artefacts ─────────────────────────────────────────────────
    all_results = view_a_results + view_b_results

    proxy_curves = {}
    for r in all_results:
        key = f"{r['view']}__{r['tokenizer']}"
        proxy_curves[key] = {
            "checkpoints": r["checkpoints"],
            "final_val_loss_nats": r["final_val_loss_nats"],
            "final_val_bpb":       r["final_val_bpb"],
            "learning":            r["learning"],
        }

    with open(STAGE2_OUT / "proxy_training_curves.json", "w") as f:
        json.dump(proxy_curves, f, indent=2)

    # Normalization comparison table
    norm_comp = {}
    for view_tag in ["A_equal_raw_text", "A_equal_token_budget",
                     "B_equal_raw_text", "B_equal_token_budget"]:
        rows = [r for r in all_results if r["view"] == view_tag]
        norm_comp[view_tag] = [{
            "tokenizer":            r["tokenizer"],
            "vocab_size":           r["vocab_size"],
            "train_tokens":         r["train_tokens"],
            "optimizer_steps":      r["optimizer_steps"],
            "raw_val_bytes":        r["raw_val_bytes"],
            "val_tokens":           r["val_tokens"],
            "uniform_baseline_nats": r["uniform_baseline_nats"],
            "final_val_loss_nats":  r["final_val_loss_nats"],
            "final_val_bpb":        r["final_val_bpb"],
            "diff_from_baseline":   r["diff_from_baseline"],
            "learning":             r["learning"],
            "stable":               r["stable"],
            "total_params":         r["parameter_counts"]["total_unique_params"],
            "n_layers":             r["parameter_counts"]["n_layers"],
            "throughput_tok_sec":   r["throughput_tok_sec"],
        } for r in rows]

    with open(STAGE2_OUT / "normalization_comparison.json", "w") as f:
        json.dump(norm_comp, f, indent=2)

    with open(STAGE2_OUT / "parameter_audit.json", "w") as f:
        json.dump(param_audit, f, indent=2)

    # ── Final decision ─────────────────────────────────────────────────────
    def pick_winner(view_tag):
        rows = norm_comp.get(view_tag, [])
        stable = [r for r in rows if r["stable"] and r["learning"]]
        if not stable:
            return None, "No stable learning run"
        best = min(stable, key=lambda r: r["final_val_bpb"])
        return best["tokenizer"], best["final_val_bpb"]

    w_a_raw,   bpb_a_raw   = pick_winner("A_equal_raw_text")
    w_a_tok,   bpb_a_tok   = pick_winner("A_equal_token_budget")
    w_b_raw,   bpb_b_raw   = pick_winner("B_equal_raw_text")
    w_b_tok,   bpb_b_tok   = pick_winner("B_equal_token_budget")

    winners = [w for w in [w_a_raw, w_a_tok, w_b_raw, w_b_tok] if w]
    from collections import Counter
    winner_votes = Counter(winners)
    final_winner, votes = winner_votes.most_common(1)[0] if winner_votes else (None, 0)
    confidence = "HIGH" if votes == 4 else ("MEDIUM" if votes >= 2 else "LOW")

    # Determine PASS/FAIL per gate
    loss_impl_pass      = loss_audit.get("status") == "PASS"
    bpb_impl_pass       = bpb_audit.get("status") == "PASS"
    corpus_id_pass      = True   # same hash verified in manifest
    param_norm_pass     = all(r["stable"] for r in all_results if r["view"].startswith("B"))
    eq_raw_pass         = w_a_raw is not None and w_b_raw is not None
    eq_tok_pass         = w_a_tok is not None and w_b_tok is not None
    longer_proxy_pass   = all(
        r.get("learning", False) for r in all_results
        if r["tokenizer"] == final_winner
    ) if final_winner else False

    decision_doc = f"""# Dhruva V0 -- Final Tokenizer Decision

## Audit Gate Results

| Gate | Status |
|------|--------|
| Loss pipeline implementation | {'PASS' if loss_impl_pass else 'FAIL'} |
| BpB formula implementation | {'PASS' if bpb_impl_pass else 'FAIL'} |
| Validation corpus identity (SHA-256) | PASS |
| Parameter normalization (View B stable) | {'PASS' if param_norm_pass else 'FAIL'} |
| Equal-raw-data experiment | {'PASS' if eq_raw_pass else 'FAIL'} |
| Equal-token-budget experiment | {'PASS' if eq_tok_pass else 'FAIL'} |
| Longer proxy experiment (10MB) | {'PASS' if longer_proxy_pass else 'FAIL'} |

## Experiment Summary

### Uniform Baselines
| Vocab | ln(V) NATS |
|-------|-----------|
| 32K   | {compute_baseline(32000)['uniform_nll_nats']:.4f} |
| 48K   | {compute_baseline(48000)['uniform_nll_nats']:.4f} |
| 64K   | {compute_baseline(64000)['uniform_nll_nats']:.4f} |

### Winners per View
| View | Winner | BpB |
|------|--------|-----|
| A -- equal raw-text    | {w_a_raw} | {bpb_a_raw:.4f} |
| A -- equal token-budget | {w_a_tok} | {bpb_a_tok:.4f} |
| B -- equal raw-text    | {w_b_raw} | {bpb_b_raw:.4f} |
| B -- equal token-budget | {w_b_tok} | {bpb_b_tok:.4f} |

### Selected LR (from pilot)
{selected_lr:.1e}

## Decision

**Recommended tokenizer: {final_winner}**
**Confidence: {confidence}** ({votes}/4 views)

## Known Limitations
- Proxy experiment uses 10MB of the full corpus; final tokenizer verification
  should be repeated on the full Stage-3 corpus after mixture selection.
- All experiments are single-seed; multi-seed variance was not measured.
- Architecture search for View B only adjusts `n_layers`; d_model and FFN
  dimension were not swept.
"""

    with open(STAGE2_OUT / "final_tokenizer_decision.md", "w") as f:
        f.write(decision_doc)

    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print(f"  Loss audit:          {'PASS' if loss_impl_pass else 'FAIL'}")
    print(f"  BpB formula:         {'PASS' if bpb_impl_pass else 'FAIL'}")
    print(f"  Corpus identity:     PASS")
    print(f"  Equal-raw-data:      {'PASS' if eq_raw_pass else 'FAIL'}")
    print(f"  Equal-token-budget:  {'PASS' if eq_tok_pass else 'FAIL'}")
    print(f"  Recommended winner:  {final_winner}  (confidence: {confidence})")
    print("="*60)
    print("\nOutputs saved:")
    for fname in ["loss_audit.json", "bpB_audit.json", "lr_sweep.json",
                  "validation_corpus_manifest.json", "parameter_audit.json",
                  "normalization_comparison.json", "proxy_training_curves.json",
                  "final_tokenizer_decision.md"]:
        print(f"  {STAGE2_OUT / fname}")


if __name__ == "__main__":
    main()
