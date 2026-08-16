"""
tests/unit/test_tokenizer_audit.py
Dhruva V0 — Tokenizer Unit Tests

Covers:
  - BOS/EOS/PAD token presence
  - Encode/decode round-trip
  - Multilingual round-trip (English, Bengali, Hindi, Code)
  - Vocab size consistency
  - Tokenizer/model vocab dimension match
  - Deterministic encoding
  - Special token handling
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.core.model.config import ModelConfig
from myllm.core.model.transformer import MyLLMModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_tokenizer() -> str:
    """Find the best available tokenizer for testing."""
    from myllm.utils.env import get_project_root
    root = get_project_root()
    candidates = [
        str(root / "output/dhruva_100m/tokenizer"),
        str(root / "output/v0_100m/final_model/tokenizer"),
        str(root / "output/v0_100m/tokenizer"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


TOKENIZER_PATH = find_tokenizer()
SKIP_IF_NO_TOKENIZER = pytest.mark.skipif(
    TOKENIZER_PATH is None,
    reason="No trained tokenizer found. Run build_tokenizer_sample.py first."
)


# ── Test 1: Special Tokens ────────────────────────────────────────────────────

@SKIP_IF_NO_TOKENIZER
def test_special_tokens_exist():
    """BOS and EOS tokens must have defined IDs."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.bos_token_id is not None, "BOS token ID is None"
    assert tok.eos_token_id is not None, "EOS token ID is None"
    assert tok.bos_token_id != tok.eos_token_id, "BOS and EOS must be different IDs"


@SKIP_IF_NO_TOKENIZER
def test_special_tokens_in_vocab():
    """Special token IDs must be within vocab range."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    vs = tok.vocab_size
    assert 0 <= tok.bos_token_id < vs, f"BOS ID {tok.bos_token_id} out of range [0, {vs})"
    assert 0 <= tok.eos_token_id < vs, f"EOS ID {tok.eos_token_id} out of range [0, {vs})"


# ── Test 2: Encode/Decode Round-Trip ─────────────────────────────────────────

ROUND_TRIP_SAMPLES = [
    # English
    ("english", "The quick brown fox jumps over the lazy dog."),
    # Bengali
    ("bengali", "আমি বাংলায় কথা বলি।"),
    # Hindi
    ("hindi", "मैं हिंदी बोलता हूँ।"),
    # Tamil
    ("tamil", "நான் தமிழ் பேசுகிறேன்."),
    # Arabic
    ("arabic", "مرحبا بالعالم"),
    # Code
    ("code", "def hello_world():\n    print('Hello, World!')\n"),
    # Mixed
    ("mixed", "Hello नमस्ते こんにちは 안녕하세요"),
    # Numbers
    ("numbers", "1234567890 3.14159 -42"),
]


@SKIP_IF_NO_TOKENIZER
def test_round_trip_english():
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert len(ids) > 0, "No tokens produced for English text"
    assert decoded.strip() == text.strip(), f"Round-trip failed:\n  Original: {repr(text)}\n  Decoded:  {repr(decoded)}"


@SKIP_IF_NO_TOKENIZER
def test_round_trip_bengali():
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "আমি বাংলায় কথা বলি।"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert len(ids) > 0, "No tokens produced for Bengali text"
    # Accept minor whitespace normalization
    assert decoded.strip() == text.strip(), f"Bengali round-trip failed:\n  {repr(text)}\n  {repr(decoded)}"


@SKIP_IF_NO_TOKENIZER
def test_round_trip_hindi():
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "मैं हिंदी बोलता हूँ।"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert len(ids) > 0, "No tokens produced for Hindi text"
    assert decoded.strip() == text.strip(), f"Hindi round-trip failed:\n  {repr(text)}\n  {repr(decoded)}"


@SKIP_IF_NO_TOKENIZER
def test_round_trip_code():
    """
    Code round-trip test.

    KNOWN BEHAVIOR (SentencePiece): The decoder always prepends a leading space
    to the first token. This is a property of SentencePiece BPE's byte-level
    encoding. Callers (generate, inference) MUST strip() the output.

    This is NOT a correctness bug in the tokenizer algorithm itself,
    but IS a documented interface hazard that callers must handle.
    Verified: inference.py generate() returns prompt+generated without stripping.
    ACTION REQUIRED: Fix inference.py to strip leading space from generated output.
    """
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "def hello_world():\n    print('Hello, World!')\n"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert len(ids) > 0, "No tokens produced for code"
    # Accept leading space (known SentencePiece behavior) - compare stripped
    assert decoded.strip() == text.strip(), (
        f"Code round-trip failed even after strip():\n"
        f"  Original: {repr(text)}\n"
        f"  Decoded:  {repr(decoded)}"
    )
    # Document the leading space hazard
    if decoded.startswith(' ') and not text.startswith(' '):
        import warnings
        warnings.warn(
            "KNOWN ISSUE: SentencePiece decode prepends a leading space. "
            "Callers MUST call .strip() on decoded output. "
            "Check inference.py generate() and generate_stream().",
            UserWarning
        )


# ── Test 3: Vocab Size Match ──────────────────────────────────────────────────

@SKIP_IF_NO_TOKENIZER
def test_vocab_size_matches_config():
    """
    tokenizer.vocab_size must match ModelConfig.vocab_size.
    A mismatch here will cause a silent embedding lookup crash or wrong loss.
    """
    tok = BPETokenizer.load(TOKENIZER_PATH)
    config = ModelConfig()  # defaults to vocab_size=32000

    assert tok.vocab_size == config.vocab_size, (
        f"CRITICAL MISMATCH: tokenizer.vocab_size={tok.vocab_size} "
        f"!= config.vocab_size={config.vocab_size}. "
        f"This will cause index out of bounds errors during training."
    )


@SKIP_IF_NO_TOKENIZER
def test_vocab_size_matches_model_embedding():
    """
    tokenizer.vocab_size must match model embedding dimension.
    Do NOT silently resize embeddings.
    """
    tok = BPETokenizer.load(TOKENIZER_PATH)
    config = ModelConfig(vocab_size=tok.vocab_size)
    import torch
    torch.manual_seed(0)
    model = MyLLMModel(config)

    assert model.token_embedding.num_embeddings == tok.vocab_size, (
        f"Model embedding dim {model.token_embedding.num_embeddings} "
        f"!= tokenizer vocab_size {tok.vocab_size}"
    )
    assert model.output_proj.out_features == tok.vocab_size, (
        f"LM head out_features {model.output_proj.out_features} "
        f"!= tokenizer vocab_size {tok.vocab_size}"
    )


# ── Test 4: Determinism ───────────────────────────────────────────────────────

@SKIP_IF_NO_TOKENIZER
def test_encoding_is_deterministic():
    """Same text must encode to same token IDs every call."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "Hello, world! This is a test of tokenizer determinism."
    ids1 = tok.encode(text, add_special_tokens=False)
    ids2 = tok.encode(text, add_special_tokens=False)
    assert ids1 == ids2, "Tokenizer encoding is non-deterministic!"


# ── Test 5: EOS Token Handling ────────────────────────────────────────────────

@SKIP_IF_NO_TOKENIZER
def test_eos_not_in_middle_of_text():
    """EOS should not appear in the middle of a normal text encoding."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "This is a normal sentence without any end of sequence markers."
    ids = tok.encode(text, add_special_tokens=False)
    # EOS should not appear mid-sequence in a normal encoding
    eos = tok.eos_token_id
    if eos is not None and eos in ids:
        mid_eos = [i for i, x in enumerate(ids) if x == eos and i < len(ids) - 1]
        assert len(mid_eos) == 0, \
            f"EOS token found mid-sequence at positions {mid_eos}. " \
            f"This suggests EOS is being incorrectly included in normal text encoding."


# ── Test 6: Token ID Range ────────────────────────────────────────────────────

@SKIP_IF_NO_TOKENIZER
def test_all_token_ids_in_range():
    """All encoded token IDs must be within [0, vocab_size)."""
    tok = BPETokenizer.load(TOKENIZER_PATH)
    text = "The quick brown fox jumps over the lazy dog. 1234. मैं।"
    ids = tok.encode(text, add_special_tokens=True)
    bad = [i for i in ids if not (0 <= i < tok.vocab_size)]
    assert len(bad) == 0, f"Out-of-range token IDs: {bad}"


if __name__ == "__main__":
    tests = [
        test_special_tokens_exist,
        test_special_tokens_in_vocab,
        test_round_trip_english,
        test_round_trip_bengali,
        test_round_trip_hindi,
        test_round_trip_code,
        test_vocab_size_matches_config,
        test_vocab_size_matches_model_embedding,
        test_encoding_is_deterministic,
        test_eos_not_in_middle_of_text,
        test_all_token_ids_in_range,
    ]

    failures = []
    skipped = []
    for test_fn in tests:
        # Check skip marker
        skip_marker = getattr(test_fn, 'pytestmark', None)
        if TOKENIZER_PATH is None:
            skipped.append(test_fn.__name__)
            print(f"  SKIP  {test_fn.__name__} (no tokenizer found)")
            continue
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
    print(f"Results: {len(tests) - len(failures) - len(skipped)}/{len(tests)} passed "
          f"({len(skipped)} skipped, {len(failures)} failed)")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        import sys; sys.exit(1)
    else:
        print("ALL TESTS PASSED (or skipped)")
