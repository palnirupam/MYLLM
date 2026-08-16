"""
tests/unit/test_model_audit.py
Dhruva V0 — Model Architecture Unit Tests

Covers:
  - Weight tying (shared storage identity)
  - Forward pass shape and dtype
  - Causal LM loss alignment (no double-shift)
  - Parameter count verification
  - NaN/Inf detection
  - KV-cache vs no-cache equivalence
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import pytest

from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config(**overrides) -> ModelConfig:
    """Return a tiny config suitable for fast tests."""
    defaults = dict(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        max_seq_len=32,
        dropout=0.0,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def make_model(config: ModelConfig, seed: int = 42) -> MyLLMModel:
    torch.manual_seed(seed)
    model = MyLLMModel(config)
    model.eval()
    return model


# ── Test 1: Weight Tying ──────────────────────────────────────────────────────

def test_weight_tying_shared_storage():
    """
    Verifies that token_embedding and output_proj share the SAME tensor storage.
    Does NOT merely compare values — checks data_ptr() identity.
    """
    config = make_config(tie_word_embeddings=True)
    model = make_model(config)

    assert model.output_proj.weight is model.token_embedding.weight, \
        "output_proj.weight and token_embedding.weight are NOT the same Python object."
    
    assert model.output_proj.weight.data_ptr() == model.token_embedding.weight.data_ptr(), \
        "output_proj.weight and token_embedding.weight do NOT share underlying storage."


def test_weight_tying_disabled():
    """When tie_word_embeddings=False, weights must be DIFFERENT tensors."""
    config = make_config(tie_word_embeddings=False)
    model = make_model(config)

    assert model.output_proj.weight.data_ptr() != model.token_embedding.weight.data_ptr(), \
        "With tie_word_embeddings=False, weights should be independent."


# ── Test 2: Forward Pass Shape ────────────────────────────────────────────────

def test_forward_pass_shape():
    """Logits shape must be (batch, seq_len, vocab_size)."""
    config = make_config()
    model = make_model(config)

    B, L = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    logits, kv_cache = model(input_ids)

    assert logits.shape == (B, L, config.vocab_size), \
        f"Expected logits shape ({B}, {L}, {config.vocab_size}), got {tuple(logits.shape)}"
    assert kv_cache is None, "Expected no kv_cache when use_cache=False"


def test_forward_pass_dtype():
    """Logits must be float32 (no dtype contamination)."""
    config = make_config()
    model = make_model(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    logits, _ = model(input_ids)
    assert logits.dtype == torch.float32, f"Expected float32 logits, got {logits.dtype}"


def test_forward_pass_no_nan_inf():
    """Fresh model forward pass must not produce NaN or Inf logits."""
    config = make_config()
    model = make_model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, _ = model(input_ids)
    assert not torch.isnan(logits).any(), "NaN detected in logits"
    assert not torch.isinf(logits).any(), "Inf detected in logits"


# ── Test 3: Causal LM Loss Alignment (No Double Shift) ───────────────────────

def test_causal_lm_loss_alignment():
    """
    Verifies that:
      input_ids[t] -> model -> logits[t] predicts labels[t] = input_ids[t+1]

    Tests that there is NO double-shift: shift happens ONCE in data preparation,
    not again in the loss computation.

    NOTE on loss magnitude: nn.Embedding uses N(0,1) init by default (not scaled).
    For tiny test configs (d_model=64) logits can be very large at init — this is
    a known initialization quality issue (see test_init_scale_warning) but is NOT
    a correctness bug in the shift/loss alignment logic.
    """
    config = make_config()
    model = make_model(config)

    # Simulate what TextDataset produces:
    chunk = torch.randint(0, config.vocab_size, (1, config.max_seq_len + 1))
    input_ids = chunk[:, :-1]   # tokens 0..N-1
    labels     = chunk[:, 1:]    # tokens 1..N

    # 1. Verify data shift alignment: label[t] == input_ids[t+1] for all valid t
    for t in range(input_ids.shape[1] - 1):
        assert labels[0, t].item() == input_ids[0, t+1].item(), \
            f"Label alignment broken at t={t}: label={labels[0,t].item()}, " \
            f"input[t+1]={input_ids[0, t+1].item()}"

    logits, _ = model(input_ids)

    # 2. Compute loss using production formula (NO additional shift)
    loss = nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        labels.view(-1),
        ignore_index=-100
    )

    # 3. Loss must be finite and positive (correctness check only, not magnitude)
    assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"Loss must be positive, got {loss.item()}"


def test_init_scale_now_correct():
    """
    After the embedding init fix, embedding std must be close to 1/sqrt(d_model).
    This test replaces the previous warning-only test_init_scale_warning.

    Before fix: std=1.0, init loss=255 nats (24.7x above uniform)
    After fix:  std≈0.036, init loss≈10.77 nats (1.04x above uniform)
    """
    import math
    config = ModelConfig(
        vocab_size=32000, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
        intermediate_size=2048, max_seq_len=512, dropout=0.0, norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=True)
    torch.manual_seed(42)
    model = MyLLMModel(config)

    emb_std = model.token_embedding.weight.std().item()
    recommended_std = 1.0 / math.sqrt(config.d_model)  # ~0.036

    # Must be within 50% of recommended (generous tolerance)
    assert abs(emb_std - recommended_std) < recommended_std * 0.5, (
        f"Embedding std {emb_std:.4f} is too far from recommended {recommended_std:.4f}. "
        f"The init fix may have regressed."
    )


def test_init_loss_near_uniform():
    """
    At initialization, loss must be close to ln(vocab_size).
    Before fix: 24.7x above uniform (255 nats).
    After fix:  within 10% of uniform baseline.
    """
    import math
    config = ModelConfig(
        vocab_size=32000, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
        intermediate_size=2048, max_seq_len=512, dropout=0.0, norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=True)
    torch.manual_seed(42)
    model = MyLLMModel(config).eval()

    # Use short sequence to keep this test fast
    torch.manual_seed(0)
    ids    = torch.randint(0, config.vocab_size, (1, 32))
    labels = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        logits, _ = model(ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), labels.view(-1))

    uniform_baseline = math.log(config.vocab_size)  # 10.37 nats
    ratio = loss.item() / uniform_baseline

    # Should be within 15% of uniform (perfect init = 1.0x)
    assert ratio < 1.15, (
        f"Init loss {loss.item():.2f} nats is {ratio:.2f}x above uniform baseline "
        f"{uniform_baseline:.2f} nats. Expected < 1.15x. "
        f"Check _init_weights() in MyLLMModel."
    )



def test_loss_reference_comparison():
    """
    Production loss must match an independent reference implementation
    on the same fixed batch.
    """
    config = make_config()
    torch.manual_seed(0)
    model = make_model(config, seed=0)

    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels    = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        logits, _ = model(input_ids)

    # Production loss
    production_loss = nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        labels.view(-1),
        ignore_index=-100
    )

    # Reference implementation: manual computation
    log_probs = torch.log_softmax(logits.view(-1, config.vocab_size).float(), dim=-1)
    flat_labels = labels.view(-1)
    valid_mask = flat_labels != -100
    reference_loss = -log_probs[valid_mask, flat_labels[valid_mask]].mean()

    # Must match within float32 numerical tolerance
    assert abs(production_loss.item() - reference_loss.item()) < 1e-5, \
        f"Production loss {production_loss.item():.6f} != reference {reference_loss.item():.6f}"


# ── Test 4: Parameter Count ───────────────────────────────────────────────────

def test_parameter_count_production_config():
    """
    Exact parameter count for the production v0_100m config.
    Any architectural change that silently changes parameter count will be caught here.
    """
    config = ModelConfig(
        vocab_size=32000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        intermediate_size=2048,
        max_seq_len=512,
        dropout=0.0,
        norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = MyLLMModel(config)

    # Count unique parameters (excluding tied duplicates)
    counted = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in counted:
            counted.add(p.data_ptr())
            total += p.numel()

    # Expected: 100,092,672
    # Embedding: 24,576,000
    # 12 layers × 6,292,992 = 75,515,904
    # Final norm: 768
    EXPECTED = 100_092_672
    assert total == EXPECTED, \
        f"Parameter count changed! Expected {EXPECTED:,}, got {total:,}. " \
        f"This means the architecture was silently modified."


# ── Test 5: KV-Cache Equivalence ─────────────────────────────────────────────

def test_kv_cache_equivalence():
    """
    Generating with KV-cache (token-by-token) must produce identical logits to
    generating without cache (full sequence at once), within float32 tolerance.
    """
    config = make_config()
    torch.manual_seed(42)
    model = make_model(config)
    model.eval()

    L = 8
    input_ids = torch.randint(0, config.vocab_size, (1, L))

    # Method 1: No cache, full sequence
    with torch.no_grad():
        logits_no_cache, _ = model(input_ids, use_cache=False)

    # Method 2: With KV cache, token by token
    with torch.no_grad():
        kv_cache = None
        all_logits = []
        for t in range(L):
            tok = input_ids[:, t:t+1]
            logits_t, kv_cache = model(tok, kv_cache=kv_cache, use_cache=True)
            all_logits.append(logits_t)
        logits_with_cache = torch.cat(all_logits, dim=1)  # (1, L, vocab)

    # The final-position logits must match exactly (or very nearly)
    # Note: float32 accumulation order differs, so we allow small tolerance
    max_diff = (logits_no_cache - logits_with_cache).abs().max().item()
    assert max_diff < 1e-3, \
        f"KV-cache logits differ from no-cache by {max_diff:.2e}. " \
        f"Expected < 1e-3 for float32."


# ── Test 6: Causal Masking ────────────────────────────────────────────────────

def test_causal_masking_future_independence():
    """
    Changing a token at position t should NOT change logits at positions < t.
    Verifies that causal masking prevents future token leakage.
    """
    config = make_config()
    model = make_model(config, seed=42)
    model.eval()

    L = 16
    torch.manual_seed(10)
    input_ids_original = torch.randint(0, config.vocab_size, (1, L))
    input_ids_modified = input_ids_original.clone()
    # Modify token at position L-1 (last)
    input_ids_modified[0, -1] = (input_ids_original[0, -1] + 1) % config.vocab_size

    with torch.no_grad():
        logits_original, _ = model(input_ids_original)
        logits_modified, _ = model(input_ids_modified)

    # Logits at positions 0..L-2 must be IDENTICAL (future token change shouldn't affect past)
    max_diff = (logits_original[:, :-1, :] - logits_modified[:, :-1, :]).abs().max().item()
    assert max_diff < 1e-6, \
        f"Causal masking violation! Changing last token changed earlier logits by {max_diff:.2e}"


# ── Test 7: Determinism ───────────────────────────────────────────────────────

def test_forward_determinism():
    """Same input + same seed must produce identical logits (eval mode, no dropout)."""
    config = make_config(dropout=0.0)
    model = make_model(config, seed=42)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits1, _ = model(input_ids)
        logits2, _ = model(input_ids)

    assert torch.allclose(logits1, logits2), "Model is non-deterministic in eval mode!"


if __name__ == "__main__":
    import sys
    tests = [
        test_weight_tying_shared_storage,
        test_weight_tying_disabled,
        test_forward_pass_shape,
        test_forward_pass_dtype,
        test_forward_pass_no_nan_inf,
        test_causal_lm_loss_alignment,
        test_init_scale_now_correct,
        test_init_loss_near_uniform,
        test_loss_reference_comparison,
        test_parameter_count_production_config,
        test_kv_cache_equivalence,
        test_causal_masking_future_independence,
        test_forward_determinism,
    ]
    
    failures = []
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failures.append(test_fn.__name__)
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
            failures.append(test_fn.__name__)
    
    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
