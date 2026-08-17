"""
myllm.training.data.fast_counter — High-Performance Resumable Token Counter.

Features:
- Batched Rust multi-threaded token counting via tokenizers.encode_batch.
- Zero whole-file RAM loading (streams lines in configurable batch chunks).
- Resumable checkpointing (resumes from byte/line offset without double-counting).
- Continuous real-time throughput metrics (docs/sec, tokens/sec, MB/sec, ETA).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional


def measure_corpus_metrics_fast(
    fpath: Path,
    bpe_tokenizer,
    batch_size: int = 2000,
    checkpoint_interval_docs: int = 10000,
    log_interval_sec: float = 3.0,
) -> Tuple[int, int, Dict[str, int], Dict[str, int], Dict[str, int], int]:
    """
    Measures exact token counts, UTF-8 byte counts, and category distributions for a JSONL file.

    Returns:
        (total_tokens, total_bytes, by_language, by_domain, by_source, total_docs)
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"Corpus file not found: {fpath}")

    file_size_bytes = fpath.stat().st_size
    ckpt_path = fpath.parent / f".checkpoint_counting_{fpath.stem}.json"

    # State tracking
    tot_tokens = 0
    tot_bytes = 0
    tot_docs = 0
    by_lang = {}
    by_dom = {}
    by_src = {}
    last_line_offset = 0

    # 1. Resume Checkpoint Loading
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r", encoding="utf-8") as cf:
                state = json.load(cf)
            if state.get("file_size_bytes") == file_size_bytes:
                tot_tokens = state.get("tot_tokens", 0)
                tot_bytes = state.get("tot_bytes", 0)
                tot_docs = state.get("tot_docs", 0)
                by_lang = state.get("by_lang", {})
                by_dom = state.get("by_dom", {})
                by_src = state.get("by_src", {})
                last_line_offset = state.get("line_offset", 0)
                print(f"  [Resume] Loaded counting checkpoint at doc {tot_docs:,} ({tot_tokens:,} tokens counted so far)")
        except Exception as e:
            print(f"  [Warning] Could not restore checkpoint: {e}. Starting from offset 0.")

    start_time = time.time()
    last_log_time = start_time
    last_ckpt_docs = tot_docs

    current_line_idx = 0
    batch_texts = []
    batch_meta = []
    batch_bytes = 0

    print(f"  [Start Counting] File: {fpath.name} ({file_size_bytes / (1024**2):.1f} MB) | Batch Size: {batch_size:,}")

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            current_line_idx += 1
            if current_line_idx <= last_line_offset:
                continue  # Skip already counted lines

            if not line.strip():
                continue

            line_b = len(line.encode("utf-8"))
            tot_bytes += line_b
            batch_bytes += line_b

            try:
                item = json.loads(line)
            except Exception:
                continue

            batch_texts.append(item.get("text", ""))
            batch_meta.append((item.get("language", "Unknown"), item.get("domain", "Unknown"), item.get("source", "Unknown")))

            # Process when batch is full
            if len(batch_texts) >= batch_size:
                # Fast Rust multi-threaded token counting
                token_counts = bpe_tokenizer.count_tokens_batch(batch_texts)

                for (lang, dom, src), count in zip(batch_meta, token_counts):
                    tot_tokens += count
                    by_lang[lang] = by_lang.get(lang, 0) + count
                    by_dom[dom] = by_dom.get(dom, 0) + count
                    by_src[src] = by_src.get(src, 0) + count

                tot_docs += len(batch_texts)
                batch_texts.clear()
                batch_meta.clear()
                batch_bytes = 0

                now = time.time()
                # Continuous progress logging
                if now - last_log_time >= log_interval_sec:
                    elapsed = max(0.001, now - start_time)
                    docs_sec = (tot_docs - last_line_offset) / elapsed
                    toks_sec = tot_tokens / elapsed
                    mb_proc = tot_bytes / (1024 * 1024)
                    tot_mb = file_size_bytes / (1024 * 1024)
                    progress_pct = (tot_bytes / max(1, file_size_bytes)) * 100.0
                    remaining_bytes = max(0, file_size_bytes - tot_bytes)
                    bytes_per_sec = (tot_bytes - state.get("tot_bytes", 0) if last_line_offset > 0 else tot_bytes) / elapsed
                    eta_sec = (remaining_bytes / max(1.0, bytes_per_sec)) if bytes_per_sec > 0 else 0

                    print(
                        f"  [{fpath.stem}] {tot_docs:7,d} docs ({progress_pct:5.1f}%) | "
                        f"Tokens: {tot_tokens:10,d} | "
                        f"Speed: {docs_sec:6.0f} docs/s ({toks_sec:8.0f} tok/s) | "
                        f"{mb_proc:5.1f}/{tot_mb:.1f} MB | "
                        f"ETA: {eta_sec:.0f}s",
                        flush=True,
                    )
                    last_log_time = now

                # Periodic checkpoint persistence
                if tot_docs - last_ckpt_docs >= checkpoint_interval_docs:
                    ckpt_data = {
                        "file_size_bytes": file_size_bytes,
                        "line_offset": current_line_idx,
                        "tot_tokens": tot_tokens,
                        "tot_bytes": tot_bytes,
                        "tot_docs": tot_docs,
                        "by_lang": by_lang,
                        "by_dom": by_dom,
                        "by_src": by_src,
                        "updated_at": time.time(),
                    }
                    try:
                        with open(ckpt_path, "w", encoding="utf-8") as cf:
                            json.dump(ckpt_data, cf)
                        last_ckpt_docs = tot_docs
                    except Exception:
                        pass

        # Final remaining batch
        if batch_texts:
            token_counts = bpe_tokenizer.count_tokens_batch(batch_texts)
            for (lang, dom, src), count in zip(batch_meta, token_counts):
                tot_tokens += count
                by_lang[lang] = by_lang.get(lang, 0) + count
                by_dom[dom] = by_dom.get(dom, 0) + count
                by_src[src] = by_src.get(src, 0) + count
            tot_docs += len(batch_texts)
            batch_texts.clear()
            batch_meta.clear()

    # Clean up checkpoint upon complete success
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass

    total_time = max(0.001, time.time() - start_time)
    print(
        f"  [Complete {fpath.stem}] Total Docs: {tot_docs:,} | Total Tokens: {tot_tokens:,} | "
        f"Total Bytes: {tot_bytes:,} | Time: {total_time:.2f}s ({tot_docs / total_time:.0f} docs/s, {tot_tokens / total_time:.0f} tok/s)\n"
    )

    return tot_tokens, tot_bytes, by_lang, by_dom, by_src, tot_docs
