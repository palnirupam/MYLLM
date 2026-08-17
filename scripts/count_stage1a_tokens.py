"""
scripts/count_stage1a_tokens.py — Fast Standalone Resumable Token Counter for Step 3.

Runs ONLY the Step 3 exact counting & manifest generation pipeline on an existing
generated corpus and frozen tokenizer without re-streaming or re-training.

Features:
- Rust-native multi-threaded batch tokenization (`encode_batch`).
- Resumable checkpointing (`.checkpoint_counting_<split>.json`).
- Live progress reporting (docs/sec, tok/sec, MB/sec, ETA).
- Updates manifests/corpus_manifest.json and manifests/asset_checksums.sha256 with exact numbers.

Usage:
  python scripts/count_stage1a_tokens.py \\
    --assets-dir /kaggle/working/dhruva-v1-assets \\
    --batch-size 2000
"""

import sys
import json
import time
import hashlib
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.training.data.fast_counter import measure_corpus_metrics_fast


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_stage1a_assets(
    assets_dir: str = "dhruva-v1-assets",
    batch_size: int = 2000,
    min_train_tokens: int = 90_000_000,
) -> dict:
    root = Path(assets_dir)
    tok_dir = root / "tokenizer"
    corpus_dir = root / "corpus"
    manifest_dir = root / "manifests"

    print("============================================================")
    print(f" DHRUVA STAGE 1A FAST TOKEN COUNTING (STEP 3 ONLY)")
    print(f" Asset Directory : {root.resolve()}")
    print(f" Batch Size      : {batch_size:,} docs/batch")
    print("============================================================")

    # 1. Load Frozen Tokenizer
    tok_file = tok_dir / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"Frozen tokenizer not found: {tok_file}")

    print(f"\n[1/3] Loading Frozen Tokenizer...")
    bpe_tokenizer = BPETokenizer.load(str(tok_dir))
    tok_sha256 = calculate_sha256(tok_file)
    print(f"  -> Vocab Size : {bpe_tokenizer.vocab_size:,}")
    print(f"  -> SHA256     : {tok_sha256}")

    # 2. Fast Batched Resumable Counting
    train_path = corpus_dir / "stage1a_train.jsonl"
    val_path = corpus_dir / "stage1a_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation corpus not found: {val_path}")

    print(f"\n[2/3] Counting Training Corpus (Batched Rust)...")
    train_tokens, train_bytes, train_lang, train_dom, train_src, train_docs = measure_corpus_metrics_fast(
        train_path,
        bpe_tokenizer,
        batch_size=batch_size,
        checkpoint_interval_docs=10000,
        log_interval_sec=3.0,
    )

    print(f"\n[2/3] Counting Validation Corpus (Batched Rust)...")
    val_tokens, val_bytes, val_lang, val_dom, val_src, val_docs = measure_corpus_metrics_fast(
        val_path,
        bpe_tokenizer,
        batch_size=batch_size,
        checkpoint_interval_docs=5000,
        log_interval_sec=3.0,
    )

    train_sha256 = calculate_sha256(train_path)
    val_sha256 = calculate_sha256(val_path)

    # 3. Update Manifests and Checksums
    print(f"\n[3/3] Updating Manifests & Master Checksums...")
    manifest_f = manifest_dir / "corpus_manifest.json"
    manifest_data = {}
    if manifest_f.exists():
        try:
            with open(manifest_f, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
        except Exception:
            manifest_data = {}

    manifest_data.update({
        "dataset_name": manifest_data.get("dataset_name", "Dhruva-Stage1A-Production"),
        "version": "1.0.0",
        "normalization": "NFC",
        "updated_at": time.time(),
        "train": {
            "file": "corpus/stage1a_train.jsonl",
            "document_count": train_docs,
            "token_count": train_tokens,
            "byte_count": train_bytes,
            "sha256": train_sha256,
            "language_distribution_tokens": train_lang,
            "domain_distribution_tokens": train_dom,
            "source_distribution_tokens": train_src,
        },
        "validation": {
            "file": "corpus/stage1a_val.jsonl",
            "document_count": val_docs,
            "token_count": val_tokens,
            "byte_count": val_bytes,
            "sha256": val_sha256,
            "language_distribution_tokens": val_lang,
            "domain_distribution_tokens": val_dom,
            "source_distribution_tokens": val_src,
        },
        "tokenizer_sha256": tok_sha256,
        "tokenizer_vocab_size": bpe_tokenizer.vocab_size,
    })

    with open(manifest_f, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    # Re-calculate master checksums
    all_pkg_files = [
        tok_dir / "tokenizer.json",
        tok_dir / "tokenizer_metadata.json",
        train_path,
        val_path,
        manifest_f,
        manifest_dir / "dataset_card.json",
    ]
    checksum_lines = []
    for fp in all_pkg_files:
        if fp.exists():
            checksum_lines.append(f"{calculate_sha256(fp)}  {fp.relative_to(root).as_posix()}")

    (manifest_dir / "asset_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"============================================================")
    print(f" COUNTING COMPLETE & MANIFESTS UPDATED")
    print(f" TRAIN TOKENS : {train_tokens:,} tokens ({train_docs:,} docs, {train_bytes/(1024**2):.1f} MB)")
    print(f" VAL TOKENS   : {val_tokens:,} tokens ({val_docs:,} docs, {val_bytes/(1024**2):.1f} MB)")
    print(f" TRAIN SHA256 : {train_sha256}")
    print(f" VAL SHA256   : {val_sha256}")
    print(f"============================================================\n")

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=str, default="dhruva-v1-assets")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--min-train-tokens", type=int, default=90_000_000)
    args = parser.parse_args()

    count_stage1a_assets(
        assets_dir=args.assets_dir,
        batch_size=args.batch_size,
        min_train_tokens=args.min_train_tokens,
    )
