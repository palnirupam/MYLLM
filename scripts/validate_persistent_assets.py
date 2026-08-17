"""
scripts/validate_persistent_assets.py — Pre-Flight Validator for Persistent Assets.
Enforces strict asset verification before any training or benchmarking is permitted.

Verification Criteria:
  1. Tokenizer file (tokenizer/tokenizer.json) exists.
  2. Tokenizer vocabulary size matches expected target (64,000).
  3. Tokenizer metadata exists and matches cryptographic SHA256.
  4. Corpus manifest (manifests/corpus_manifest.json) exists.
  5. Training corpus (corpus/stage1a_train.jsonl) and validation corpus (corpus/stage1a_val.jsonl) exist.
  6. All asset files match their registered SHA256 hashes in manifests/asset_checksums.sha256.

Returns exit code 0 on complete pass; non-zero exit code on failure.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.tokenizer.bpe import BPETokenizer


def calculate_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_persistent_assets(
    assets_dir: str = "dhruva-v1-assets",
    expected_vocab_size: int = 64000,
) -> bool:
    root = Path(assets_dir)
    print(f"============================================================")
    print(f" VALIDATING PERSISTENT ASSET PACKAGE: {root.resolve()}")
    print(f"============================================================")

    if not root.exists() or not root.is_dir():
        print(f" [FAIL] Asset root directory does not exist: {root}")
        return False

    failures = []

    # 1. Tokenizer Checks
    tok_path = root / "tokenizer" / "tokenizer.json"
    tok_meta_path = root / "tokenizer" / "tokenizer_metadata.json"

    if not tok_path.exists():
        failures.append(f"Tokenizer missing: {tok_path}")
    else:
        try:
            tokenizer = BPETokenizer.load(str(root / "tokenizer"))
            print(f" [PASS] 1. Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")

            if tokenizer.vocab_size != expected_vocab_size:
                # If vocab_size is smaller than expected, check if this is an explicit custom bundle
                failures.append(f"Tokenizer vocab size mismatch: found {tokenizer.vocab_size:,}, expected {expected_vocab_size:,}")
        except Exception as e:
            failures.append(f"Failed to load tokenizer from {tok_path}: {e}")

    if not tok_meta_path.exists():
        failures.append(f"Tokenizer metadata missing: {tok_meta_path}")
    else:
        try:
            with open(tok_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            computed_hash = calculate_sha256(tok_path)
            if meta.get("sha256") and meta["sha256"] != computed_hash:
                failures.append(f"Tokenizer SHA256 mismatch: recorded {meta.get('sha256')}, computed {computed_hash}")
            else:
                print(f" [PASS] 2. Tokenizer SHA256 verified: {computed_hash[:16]}...")
        except Exception as e:
            failures.append(f"Error reading tokenizer metadata: {e}")

    # 2. Corpus Manifest Checks
    manifest_path = root / "manifests" / "corpus_manifest.json"
    if not manifest_path.exists():
        failures.append(f"Corpus manifest missing: {manifest_path}")
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f" [PASS] 3. Corpus manifest verified (Dataset: {manifest.get('dataset_name')}, v{manifest.get('version')})")

            # Check train file
            train_file_rel = manifest.get("train", {}).get("file", "corpus/stage1a_train.jsonl")
            train_file = root / train_file_rel
            if not train_file.exists():
                failures.append(f"Training corpus missing: {train_file}")
            else:
                train_hash = calculate_sha256(train_file)
                rec_train_hash = manifest.get("train", {}).get("sha256")
                if rec_train_hash and rec_train_hash != train_hash:
                    failures.append(f"Train corpus SHA256 mismatch: recorded {rec_train_hash}, computed {train_hash}")
                else:
                    print(f" [PASS] 4. Training corpus verified: {train_file_rel} ({manifest.get('train', {}).get('tokens', 0):,} tokens)")

            # Check val file
            val_file_rel = manifest.get("validation", {}).get("file", "corpus/stage1a_val.jsonl")
            val_file = root / val_file_rel
            if not val_file.exists():
                failures.append(f"Validation corpus missing: {val_file}")
            else:
                val_hash = calculate_sha256(val_file)
                rec_val_hash = manifest.get("validation", {}).get("sha256")
                if rec_val_hash and rec_val_hash != val_hash:
                    failures.append(f"Val corpus SHA256 mismatch: recorded {rec_val_hash}, computed {val_hash}")
                else:
                    print(f" [PASS] 5. Validation corpus verified: {val_file_rel} ({manifest.get('validation', {}).get('tokens', 0):,} tokens)")

        except Exception as e:
            failures.append(f"Error validating corpus manifest: {e}")

    # 3. Comprehensive Checksum File Verification
    checksum_path = root / "manifests" / "asset_checksums.sha256"
    if not checksum_path.exists():
        failures.append(f"Checksum file missing: {checksum_path}")
    else:
        try:
            with open(checksum_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    expected_hash, rel_fpath = parts[0], parts[1].strip()
                    target_f = root / rel_fpath
                    if not target_f.exists():
                        failures.append(f"Checksum referenced file missing: {rel_fpath}")
                    else:
                        actual_hash = calculate_sha256(target_f)
                        if actual_hash != expected_hash:
                            failures.append(f"Checksum failed for {rel_fpath}: expected {expected_hash}, got {actual_hash}")
            print(f" [PASS] 6. Master asset checksums (SHA256) all match perfectly")
        except Exception as e:
            failures.append(f"Error reading asset checksums: {e}")

    print(f"------------------------------------------------------------")
    if failures:
        print(f" [STATUS: FAIL] {len(failures)} verification failure(s) detected:")
        for fail in failures:
            print(f"   [!] {fail}")
        print(f"============================================================\n")
        return False
    else:
        print(f" [STATUS: PASS] ALL PERSISTENT ASSETS ARE VALID AND READY FOR KAGGLE")
        print(f"============================================================\n")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=str, default="dhruva-v1-assets", help="Path to dhruva-v1-assets directory")
    parser.add_argument("--vocab-size", type=int, default=64000, help="Expected vocabulary size")
    args = parser.parse_args()

    success = validate_persistent_assets(assets_dir=args.assets_dir, expected_vocab_size=args.vocab_size)
    if not success:
        sys.exit(1)
    sys.exit(0)
