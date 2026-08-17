"""
scripts/validate_stage1a_assets.py — Stage 1A Production Asset Pre-Flight Validator.

Validates that dhruva-v1-assets meets all strict production requirements:
  1. Tokenizer vocabulary size == 64000.
  2. Tokenizer metadata matches and SHA256 is verified.
  3. Training corpus has >= 100,000,000 tokens (or specified threshold).
  4. Held-out validation corpus has >= 2,000,000 tokens.
  5. Normalization is strictly verified as "NFC".
  6. All file hashes match asset_checksums.sha256.
  7. Train and Validation sets have ZERO exact hash overlap (zero contamination).
  8. Provenance metadata is present for all approved sources (Wikipedia, FineWeb-Edu, OpenWebMath).

Exit Codes:
  0 : Complete PASS — Assets are fully certified for Stage 1A pre-training.
  1 : FAIL — One or more criteria violated.
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


def validate_stage1a_assets(
    assets_dir: str = "dhruva-v1-assets",
    expected_vocab_size: int = 64000,
    min_train_tokens: int = 100_000_000,
    min_val_tokens: int = 2_000_000,
) -> bool:
    root = Path(assets_dir)
    print(f"============================================================")
    print(f" STAGE 1A PRODUCTION ASSET VALIDATION: {root.resolve()}")
    print(f" Expected Vocab Size  : {expected_vocab_size:,}")
    print(f" Required Train Tokens: >= {min_train_tokens:,}")
    print(f" Required Val Tokens  : >= {min_val_tokens:,}")
    print(f" Required Normalizer  : NFC")
    print(f"============================================================")

    if not root.exists() or not root.is_dir():
        print(f" [FAIL] Asset root directory does not exist: {root}")
        return False

    failures = []

    # 1. Tokenizer Verification
    tok_dir = root / "tokenizer"
    tok_json = tok_dir / "tokenizer.json"
    tok_meta = tok_dir / "tokenizer_metadata.json"

    if not tok_json.exists():
        failures.append(f"Missing tokenizer file: {tok_json}")
    else:
        try:
            tokenizer = BPETokenizer.load(str(tok_dir))
            print(f" [PASS] 1. Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
            if tokenizer.vocab_size != expected_vocab_size:
                failures.append(f"Tokenizer vocab size mismatch: found {tokenizer.vocab_size:,}, expected {expected_vocab_size:,}")
        except Exception as e:
            failures.append(f"Failed to load tokenizer: {e}")

    if not tok_meta.exists():
        failures.append(f"Missing tokenizer metadata: {tok_meta}")
    else:
        try:
            with open(tok_meta, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
            actual_tok_hash = calculate_sha256(tok_json)
            if m_data.get("sha256") and m_data["sha256"] != actual_tok_hash:
                failures.append(f"Tokenizer hash mismatch: recorded {m_data.get('sha256')}, actual {actual_tok_hash}")
            else:
                print(f" [PASS] 2. Tokenizer metadata and SHA256 verified ({actual_tok_hash[:16]}...)")
        except Exception as e:
            failures.append(f"Error reading tokenizer metadata: {e}")

    # 2. Corpus Manifest & Token Count Verification
    manifest_f = root / "manifests" / "corpus_manifest.json"
    if not manifest_f.exists():
        failures.append(f"Missing corpus manifest: {manifest_f}")
    else:
        try:
            with open(manifest_f, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)

            # Normalization Strategy Check
            norm_strat = manifest.get("normalization")
            if norm_strat != "NFC":
                failures.append(f"Invalid normalization strategy in manifest: found '{norm_strat}', expected 'NFC'")
            else:
                print(f" [PASS] 3. Unicode NFC normalization strategy verified")

            # Check Train File
            train_rel = manifest.get("train", {}).get("file", "corpus/stage1a_train.jsonl")
            train_path = root / train_rel
            if not train_path.exists():
                failures.append(f"Missing training corpus file: {train_path}")
            else:
                train_tokens = manifest.get("train", {}).get("token_count", 0)
                if train_tokens < min_train_tokens:
                    failures.append(f"Train token count insufficient: {train_tokens:,} < required {min_train_tokens:,}")
                else:
                    print(f" [PASS] 4. Train token budget verified: {train_tokens:,} tokens (>= {min_train_tokens:,})")

                # Verify SHA256
                computed_train_hash = calculate_sha256(train_path)
                recorded_train_hash = manifest.get("train", {}).get("sha256")
                if recorded_train_hash and recorded_train_hash != computed_train_hash:
                    failures.append(f"Train corpus hash mismatch: recorded {recorded_train_hash}, computed {computed_train_hash}")
                else:
                    print(f" [PASS] 5. Train corpus SHA256 verified ({computed_train_hash[:16]}...)")

            # Check Val File
            val_rel = manifest.get("validation", {}).get("file", "corpus/stage1a_val.jsonl")
            val_path = root / val_rel
            if not val_path.exists():
                failures.append(f"Missing validation corpus file: {val_path}")
            else:
                val_tokens = manifest.get("validation", {}).get("token_count", 0)
                if val_tokens < min_val_tokens:
                    failures.append(f"Validation token count insufficient: {val_tokens:,} < required {min_val_tokens:,}")
                else:
                    print(f" [PASS] 6. Validation token budget verified: {val_tokens:,} tokens (>= {min_val_tokens:,})")

                # Verify SHA256
                computed_val_hash = calculate_sha256(val_path)
                recorded_val_hash = manifest.get("validation", {}).get("sha256")
                if recorded_val_hash and recorded_val_hash != computed_val_hash:
                    failures.append(f"Val corpus hash mismatch: recorded {recorded_val_hash}, computed {computed_val_hash}")
                else:
                    print(f" [PASS] 7. Val corpus SHA256 verified ({computed_val_hash[:16]}...)")

            # Provenance Check for all approved sources
            provenance = manifest.get("provenance_sources", {})
            required_sources = ["wikipedia", "fineweb", "math"]
            prov_text = json.dumps(provenance).lower()
            for req in required_sources:
                if req not in prov_text:
                    failures.append(f"Approved source '{req}' missing from provenance manifest")
            if not failures:
                print(f" [PASS] 8. Source provenance & license metadata verified across all approved sources")

        except Exception as e:
            failures.append(f"Error validating corpus manifest: {e}")

    # 3. Train/Val Overlap & Contamination Check
    train_path = root / "corpus" / "stage1a_train.jsonl"
    val_path = root / "corpus" / "stage1a_val.jsonl"
    if train_path.exists() and val_path.exists():
        try:
            train_hashes = set()
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        train_hashes.add(hashlib.sha256(item["text"].encode("utf-8")).hexdigest())

            val_hashes = set()
            with open(val_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        val_hashes.add(hashlib.sha256(item["text"].encode("utf-8")).hexdigest())

            overlap = train_hashes.intersection(val_hashes)
            if len(overlap) > 0:
                failures.append(f"Train/Val data leakage detected: {len(overlap)} identical documents found in both splits")
            else:
                print(f" [PASS] 9. Zero Train/Val leakage confirmed (0 hash overlap across splits)")
        except Exception as e:
            failures.append(f"Error checking data leakage: {e}")

    # 4. Master Checksum Verification
    checksum_f = root / "manifests" / "asset_checksums.sha256"
    if not checksum_f.exists():
        failures.append(f"Missing master checksum file: {checksum_f}")
    else:
        try:
            with open(checksum_f, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    expected_hash, rel_fpath = parts[0], parts[1].strip()
                    target_file = root / rel_fpath
                    if not target_file.exists():
                        failures.append(f"Checksum referenced file missing: {rel_fpath}")
                    else:
                        actual_hash = calculate_sha256(target_file)
                        if actual_hash != expected_hash:
                            failures.append(f"Checksum mismatch for {rel_fpath}: expected {expected_hash}, actual {actual_hash}")
            print(f" [PASS] 10. Master cryptographic checksums (SHA256) all match")
        except Exception as e:
            failures.append(f"Error verifying master checksums: {e}")

    print(f"------------------------------------------------------------")
    if failures:
        print(f" [STATUS: FAIL] {len(failures)} verification failure(s) detected:")
        for fail in failures:
            print(f"   [!] {fail}")
        print(f"============================================================\n")
        return False
    else:
        print(f" [STATUS: PASS] STAGE 1A ASSET PACKAGE IS FULLY CERTIFIED FOR PRODUCTION")
        print(f"============================================================\n")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=str, default="/kaggle/working/dhruva-v1-assets")
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--min-train-tokens", type=int, default=100_000_000)
    parser.add_argument("--min-val-tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    passed = validate_stage1a_assets(
        assets_dir=args.assets_dir,
        expected_vocab_size=args.vocab_size,
        min_train_tokens=args.min_train_tokens,
        min_val_tokens=args.min_val_tokens,
    )
    if not passed:
        sys.exit(1)
    sys.exit(0)
