"""
myllm.training.data.sampler — Deterministic Stratified Stage 1A Corpus Sampler.

Features:
- Deterministic sampling with fixed seed (default: 20260817).
- Stratified sampling by language and domain to preserve linguistic & conceptual diversity.
- Oversampling of underrepresented Indic languages (Bengali, Hindi) up to full availability without synthetic fabrication.
- Cap on OpenWebMath to prevent mathematical tokens from dominating the general-language stage.
- Two-pass memory-efficient stream indexing (never loads entire 491M corpus into RAM).
- Strict tracking of per-source, per-language, and per-domain tokens.
- Complete train/val isolation (held-out validation corpus is untouched).
"""

import os
import sys
import json
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from myllm.core.tokenizer.bpe import BPETokenizer


DEFAULT_STAGE1A_RATIOS = {
    "Bengali": 0.15,      # 15M tokens (Indic priority)
    "Hindi": 0.15,        # 15M tokens (Indic priority)
    "English_Wikipedia": 0.15, # 15M tokens (Encyclopedic English)
    "English_Educational": 0.38, # 38M tokens (FineWeb-Edu)
    "Mathematics": 0.17,  # 17M tokens (OpenWebMath reasoning)
}


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def classify_document_category(record: Dict[str, Any]) -> str:
    """Maps document metadata to sampling stratum category."""
    lang = record.get("language", "English")
    dom = record.get("domain", "")
    src = record.get("source", "").lower()

    if "bengali" in lang.lower() or "bn" in lang.lower():
        return "Bengali"
    if "hindi" in lang.lower() or "hi" in lang.lower():
        return "Hindi"
    if "math" in dom.lower() or "math" in src:
        return "Mathematics"
    if "fineweb" in src or "educational" in dom.lower():
        return "English_Educational"
    return "English_Wikipedia"


class DeterministicStage1ASampler:
    def __init__(
        self,
        seed: int = 20260817,
        target_tokens: int = 100_000_000,
        category_ratios: Optional[Dict[str, float]] = None,
    ):
        self.seed = seed
        self.target_tokens = target_tokens
        self.category_ratios = category_ratios or DEFAULT_STAGE1A_RATIOS
        self.rng = random.Random(self.seed)

    def sample_corpus(
        self,
        input_corpus_path: str,
        output_corpus_path: str,
        bpe_tokenizer,
        batch_size: int = 5000,
    ) -> Dict[str, Any]:
        """
        Samples exactly target_tokens from input_corpus_path into output_corpus_path.
        Uses two-pass index-then-stream to minimize RAM usage.
        """
        in_path = Path(input_corpus_path)
        out_path = Path(output_corpus_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input corpus not found: {in_path}")

        print(f"============================================================")
        print(f" DHRUVA STAGE 1A DETERMINISTIC SAMPLER")
        print(f" Input Master Corpus   : {in_path.resolve()} ({in_path.stat().st_size / (1024**2):.1f} MB)")
        print(f" Output Sampled Corpus : {out_path.resolve()}")
        print(f" Target Token Budget   : {self.target_tokens:,} tokens")
        print(f" Sampling Seed         : {self.seed}")
        print(f"============================================================")

        # ------------------------------------------------------------
        # PASS 1: Lightweight Indexing & Category Bucketing
        # ------------------------------------------------------------
        print("\n>>> [PASS 1/2] Indexing Document Offsets, Categories & Token Counts...")
        t0 = time.time()

        # buckets[category] = list of (line_offset, byte_offset, token_count, source, language, domain)
        buckets: Dict[str, List[Tuple[int, int, int, str, str, str]]] = {cat: [] for cat in self.category_ratios}
        buckets["Other"] = []

        total_input_docs = 0
        total_input_tokens = 0
        total_input_bytes = 0

        with open(in_path, "r", encoding="utf-8") as f:
            line_idx = 0
            while True:
                byte_offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line_idx += 1
                if not line.strip():
                    continue

                total_input_bytes += len(line.encode("utf-8"))
                item = json.loads(line)
                cat = classify_document_category(item)
                if cat not in buckets:
                    cat = "Other"

                # If token_count is already cached in JSON record, use it; otherwise compute fast batch
                tok_cnt = item.get("token_count")
                if tok_cnt is None:
                    tok_cnt = len(bpe_tokenizer.encode(item["text"], add_special_tokens=False))

                src = item.get("source", "Unknown")
                lang = item.get("language", "Unknown")
                dom = item.get("domain", "Unknown")

                buckets[cat].append((line_idx, byte_offset, tok_cnt, src, lang, dom))
                total_input_docs += 1
                total_input_tokens += tok_cnt

                if total_input_docs % 50000 == 0:
                    print(f"  Indexed {total_input_docs:,} docs ({total_input_tokens:,} tokens)...", flush=True)

        p1_time = max(0.001, time.time() - t0)
        print(f"  Pass 1 Complete in {p1_time:.1f}s. Total Master Corpus: {total_input_docs:,} docs | {total_input_tokens:,} tokens\n")

        for cat, doc_list in buckets.items():
            cat_toks = sum(t[2] for t in doc_list)
            pct = (cat_toks / max(1, total_input_tokens)) * 100.0
            print(f"  Category [{cat:19s}]: {len(doc_list):7,d} docs | {cat_toks:11,d} tokens ({pct:5.1f}%)")

        if total_input_tokens < self.target_tokens:
            raise RuntimeError(f"Master corpus tokens ({total_input_tokens:,}) is less than target ({self.target_tokens:,})!")

        # ------------------------------------------------------------
        # Stratified Deterministic Allocation
        # ------------------------------------------------------------
        print("\n>>> Allocating Stratified Quotas...")
        selected_records_meta = []  # list of (byte_offset, token_count, src, lang, dom, cat)
        accumulated_tokens = 0
        allocated_by_cat = {}

        # First pass: Allocate per category according to ratio
        unfilled_tokens = 0

        for cat, ratio in self.category_ratios.items():
            cat_target = int(self.target_tokens * ratio)
            avail_docs = buckets.get(cat, [])
            # Deterministic shuffle within bucket using seeded RNG
            self.rng.shuffle(avail_docs)

            cat_tokens_collected = 0
            for byte_info in avail_docs:
                if cat_tokens_collected + byte_info[2] > cat_target and cat_tokens_collected > 0:
                    # Allow slight document boundary crossing if close, else stop
                    if (cat_tokens_collected + byte_info[2] - cat_target) < (cat_target - cat_tokens_collected):
                        selected_records_meta.append(byte_info)
                        cat_tokens_collected += byte_info[2]
                    break
                selected_records_meta.append(byte_info)
                cat_tokens_collected += byte_info[2]

            allocated_by_cat[cat] = cat_tokens_collected
            accumulated_tokens += cat_tokens_collected
            print(f"  Allocated [{cat:19s}]: {cat_tokens_collected:,} tokens (Target: {cat_target:,})")
            if cat_tokens_collected < cat_target:
                unfilled_tokens += (cat_target - cat_tokens_collected)

        # Second pass: If any quota was unfilled (e.g. Indic text exhausted), fill from largest high-quality bucket
        if accumulated_tokens < self.target_tokens:
            deficit = self.target_tokens - accumulated_tokens
            print(f"\n  Filling remaining deficit ({deficit:,} tokens) from available high-quality English corpus...")
            already_selected_offsets = set(b[1] for b in selected_records_meta)
            pool = buckets.get("English_Educational", []) + buckets.get("English_Wikipedia", [])
            self.rng.shuffle(pool)

            for byte_info in pool:
                if byte_info[1] in already_selected_offsets:
                    continue
                selected_records_meta.append(byte_info)
                already_selected_offsets.add(byte_info[1])
                accumulated_tokens += byte_info[2]
                if accumulated_tokens >= self.target_tokens:
                    break

        # Final Deterministic Global Shuffle of Selected Document Ordering
        print(f"\n>>> Deterministically Shuffling Selected Stream (Total Selected: {len(selected_records_meta):,} docs)...")
        self.rng.shuffle(selected_records_meta)

        # ------------------------------------------------------------
        # PASS 2: Streaming Write of Exact Sampled Corpus
        # ------------------------------------------------------------
        print("\n>>> [PASS 2/2] Writing Sampled Corpus...")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        final_train_tokens = 0
        final_train_bytes = 0
        final_train_docs = 0
        by_lang = {}
        by_dom = {}
        by_src = {}

        t1 = time.time()
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for byte_info in selected_records_meta:
                line_idx, byte_offset, tok_cnt, src, lang, dom = byte_info
                fin.seek(byte_offset)
                line = fin.readline()
                if not line.strip():
                    continue

                final_train_bytes += len(line.encode("utf-8"))
                final_train_tokens += tok_cnt
                final_train_docs += 1

                by_lang[lang] = by_lang.get(lang, 0) + tok_cnt
                by_dom[dom] = by_dom.get(dom, 0) + tok_cnt
                by_src[src] = by_src.get(src, 0) + tok_cnt

                fout.write(line)

                if final_train_docs % 25000 == 0:
                    print(f"  Written {final_train_docs:,} docs ({final_train_tokens:,} tokens)...", flush=True)

        p2_time = max(0.001, time.time() - t1)
        out_sha256 = calculate_sha256(out_path)

        # ------------------------------------------------------------
        # Summary & Manifest Generation
        # ------------------------------------------------------------
        manifest_data = {
            "requested_train_tokens": self.target_tokens,
            "actual_selected_train_tokens": final_train_tokens,
            "document_boundary_delta_tokens": final_train_tokens - self.target_tokens,
            "total_documents_selected": final_train_docs,
            "total_bytes": final_train_bytes,
            "sampling_seed": self.seed,
            "sampled_corpus_sha256": out_sha256,
            "input_master_corpus_sha256": calculate_sha256(in_path),
            "per_language_tokens": by_lang,
            "per_domain_tokens": by_dom,
            "per_source_tokens": by_src,
            "created_at": time.time(),
        }

        manifest_path = out_path.parent.parent / "manifests" / "stage1a_sampling_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)

        manifest_sha256 = calculate_sha256(manifest_path)
        manifest_data["sampler_manifest_sha256"] = manifest_sha256

        print(f"\n============================================================")
        print(f" DHRUVA STAGE 1A SAMPLING SUMMARY")
        print(f" Requested Train Tokens     : {self.target_tokens:,}")
        print(f" Actual Selected Tokens     : {final_train_tokens:,}")
        print(f" Document Boundary Delta    : {final_train_tokens - self.target_tokens:+,} tokens ({((final_train_tokens - self.target_tokens)/self.target_tokens)*100:+.3f}%)")
        print(f" Selected Documents         : {final_train_docs:,}")
        print(f" Total Selected Bytes       : {final_train_bytes:,} ({final_train_bytes/(1024**2):.1f} MB)")
        print(f" Sampling Seed              : {self.seed}")
        print(f" Sampled Corpus SHA256      : {out_sha256}")
        print(f" Sampler Manifest SHA256    : {manifest_sha256}")
        print(f"\n Per Language Tokens:")
        for l, cnt in by_lang.items():
            print(f"   - {l:15s}: {cnt:10,d} ({cnt/final_train_tokens*100:5.1f}%)")
        print(f"\n Per Domain Tokens:")
        for d, cnt in by_dom.items():
            print(f"   - {d:25s}: {cnt:10,d} ({cnt/final_train_tokens*100:5.1f}%)")
        print(f"\n Per Source Tokens:")
        for s, cnt in by_src.items():
            print(f"   - {s:35s}: {cnt:10,d} ({cnt/final_train_tokens*100:5.1f}%)")
        print(f"============================================================\n")

        return manifest_data


def sample_stage1a_corpus_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-corpus", type=str, default="/kaggle/working/dhruva-v1-assets/corpus/stage1a_train_master.jsonl")
    parser.add_argument("--output-corpus", type=str, default="/kaggle/working/dhruva-v1-assets/corpus/stage1a_train.jsonl")
    parser.add_argument("--tokenizer-dir", type=str, default="/kaggle/working/dhruva-v1-assets/tokenizer")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    tok = BPETokenizer.load(args.tokenizer_dir)
    sampler = DeterministicStage1ASampler(seed=args.seed, target_tokens=args.target_tokens)
    sampler.sample_corpus(
        input_corpus_path=args.input_corpus,
        output_corpus_path=args.output_corpus,
        bpe_tokenizer=tok,
    )


if __name__ == "__main__":
    sample_stage1a_corpus_cli()
