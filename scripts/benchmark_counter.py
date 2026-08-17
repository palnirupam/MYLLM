"""
scripts/benchmark_counter.py — Benchmark Token Counting: Sequential vs Rust Batched.
Measures throughput (docs/sec and tokens/sec) and verifies exact correctness:
  batched_count == reference_count
"""

import sys
import json
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.data.fast_counter import measure_corpus_metrics_fast


def run_counter_benchmark(num_docs: int = 10000, batch_size: int = 2000):
    print(f"============================================================")
    print(f" TOKEN COUNTING BENCHMARK: Sequential vs Rust Batched ({num_docs:,} docs)")
    print(f"============================================================")

    # 1. Load Tokenizer
    tok_dir = Path("dhruva-v1-assets/tokenizer")
    if not (tok_dir / "tokenizer.json").exists():
        tok_dir = Path("tokenizer")
    tokenizer = BPETokenizer.load(str(tok_dir))

    # 2. Generate Synthetic Benchmark JSONL
    sample_texts = [
        "The quick brown fox jumps over the lazy dog in mathematics and computer science.",
        "বাংলা ভাষায় মেশিন লার্নিং ও কৃত্রিম বুদ্ধিমত্তা নিয়ে গভীর গবেষণা চলছে।",
        "हिंदी साहित्य में आर्यभट और प्राचीन भारतीय वैज्ञानिकों का अभूतपूर्व योगदान रहा है।",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])\n",
        "Photosynthesis converts solar energy into chemical energy stored in glucose bonds.",
    ]

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as tf:
        temp_fpath = Path(tf.name)
        for i in range(num_docs):
            txt = sample_texts[i % len(sample_texts)]
            rec = {
                "doc_id": f"doc-{i:06d}",
                "text": txt,
                "language": "English" if i % 2 == 0 else "Bengali",
                "domain": "General",
                "source": "Benchmark",
            }
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        # A) Measure Reference Sequential (Old Implementation)
        print("\n[Method A] Running Sequential Single-Doc Tokenization...")
        t0 = time.time()
        ref_tokens = 0
        ref_bytes = 0
        with open(temp_fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ref_bytes += len(line.encode("utf-8"))
                item = json.loads(line)
                toks = len(tokenizer.encode(item["text"], add_special_tokens=False))
                ref_tokens += toks
        t_seq = max(0.001, time.time() - t0)
        seq_docs_sec = num_docs / t_seq
        seq_toks_sec = ref_tokens / t_seq
        print(f"  Result: {ref_tokens:,} tokens | Time: {t_seq:.3f}s | Speed: {seq_docs_sec:,.0f} docs/s ({seq_toks_sec:,.0f} tok/s)")

        # B) Measure Fast Batched Rust Implementation
        print("\n[Method B] Running Rust Batched Tokenization (batch_size=2000)...")
        t0 = time.time()
        fast_tokens, fast_bytes, by_lang, by_dom, by_src, fast_docs = measure_corpus_metrics_fast(
            temp_fpath,
            tokenizer,
            batch_size=batch_size,
            log_interval_sec=1.0,
        )
        t_fast = max(0.001, time.time() - t0)
        fast_docs_sec = num_docs / t_fast
        fast_toks_sec = fast_tokens / t_fast
        print(f"  Result: {fast_tokens:,} tokens | Time: {t_fast:.3f}s | Speed: {fast_docs_sec:,.0f} docs/s ({fast_toks_sec:,.0f} tok/s)")

        # Correctness Assertions
        speedup = t_seq / t_fast
        print(f"\n============================================================")
        print(f" BENCHMARK COMPARISON ({num_docs:,} Documents)")
        print(f" Sequential Speed : {seq_docs_sec:8,.0f} docs/s | {seq_toks_sec:10,.0f} tokens/s")
        print(f" Batched Rust Speed: {fast_docs_sec:8,.0f} docs/s | {fast_toks_sec:10,.0f} tokens/s")
        print(f" Measured Speedup : {speedup:.1f}x Faster")
        print(f" Exact Match Check:")
        print(f"   Tokens Match   : {fast_tokens == ref_tokens} ({fast_tokens:,} vs {ref_tokens:,})")
        print(f"   Bytes Match    : {fast_bytes == ref_bytes} ({fast_bytes:,} vs {ref_bytes:,})")
        print(f"============================================================")

        assert fast_tokens == ref_tokens, f"Mismatch: {fast_tokens} != {ref_tokens}"
        assert fast_bytes == ref_bytes, f"Mismatch: {fast_bytes} != {ref_bytes}"
        assert fast_docs == num_docs, f"Mismatch: {fast_docs} != {num_docs}"
        print("\n[PASS] Correctness verified: batched_count == reference_count (100% exact match)\n")

    finally:
        if temp_fpath.exists():
            temp_fpath.unlink()


if __name__ == "__main__":
    run_counter_benchmark(num_docs=10000, batch_size=2000)
