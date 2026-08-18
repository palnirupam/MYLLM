#!/usr/bin/env python3
"""Build a streaming uint32 token shard after the V2 quality gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tempfile
import os
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.core.tokenizer.bpe import BPETokenizer
from myllm.data.quality import QualityPolicy, analyze_document
from myllm.training.artifacts import sha256_file, fsync_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-manifest", required=True)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    tokenizer_dir = Path(args.tokenizer)
    output = Path(args.output)
    quality_manifest_path = Path(args.quality_manifest)
    quality_manifest = json.loads(quality_manifest_path.read_text(encoding="utf-8"))
    if quality_manifest.get("status") != "PASS":
        raise RuntimeError("quality manifest is not PASS; refusing to pack corpus")
    expected_sha = quality_manifest.get("cleaned_corpus_sha256") or quality_manifest.get("corpus_sha256")
    actual_sha = sha256_file(corpus)
    if expected_sha != actual_sha:
        raise RuntimeError("corpus SHA does not match the quality manifest")

    tokenizer = BPETokenizer.load(str(tokenizer_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing packed output: {output}")
    temp_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    token_path = temp_output / "tokens.uint32"
    token_count = 0
    documents = 0
    seen_hashes: set[str] = set()
    policy = QualityPolicy()
    language_tokens = Counter()
    domain_tokens = Counter()

    with corpus.open("r", encoding="utf-8") as source, token_path.open("wb") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            result = analyze_document(record, policy)
            if not result.accepted:
                raise RuntimeError(f"quality failure at corpus line {line_number}: {result.reasons}")
            if result.content_hash in seen_hashes:
                raise RuntimeError(f"duplicate document at corpus line {line_number}")
            seen_hashes.add(result.content_hash)
            ids = tokenizer.encode(result.normalized_text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            np.asarray(ids, dtype=np.uint32).tofile(target)
            token_count += len(ids)
            language_tokens[str(record["language"])] += len(ids)
            domain_tokens[str(record["domain"])] += len(ids)
            documents += 1
        target.flush()
        os.fsync(target.fileno())

    expected_bytes = token_count * np.dtype(np.uint32).itemsize
    if token_path.stat().st_size != expected_bytes:
        raise RuntimeError("packed token byte length validation failed")
    mapped = np.memmap(token_path, dtype=np.uint32, mode="r", shape=(token_count,))
    if token_count and int(mapped.max()) >= tokenizer.vocab_size:
        raise RuntimeError("packed token ID exceeds tokenizer vocabulary")
    del mapped

    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "preprocessing_version": quality_manifest.get("preprocessing_version", "unknown"),
        "source_corpus": str(corpus),
        "source_corpus_sha256": actual_sha,
        "quality_manifest_sha256": sha256_file(quality_manifest_path),
        "tokenizer": str(tokenizer_dir),
        "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "documents": documents,
        "token_count": token_count,
        "tokens_bytes": expected_bytes,
        "tokens_sha256": sha256_file(token_path),
        "eos_token_count": documents,
        "discarded_remainder": 0,
        "language_tokens": dict(language_tokens),
        "language_token_shares": {key: count / max(1, token_count) for key, count in language_tokens.items()},
        "domain_tokens": dict(domain_tokens),
        "dtype": "uint32",
        "tokens_file": "tokens.uint32",
    }
    (temp_output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fsync_file(temp_output / "manifest.json")
    (temp_output / "COMPLETE").write_text("complete\n", encoding="ascii")
    fsync_file(temp_output / "COMPLETE")
    temp_output.rename(output)
    print("DHRUVA V2 PACKED CORPUS: PASS")
    print(f"Documents : {documents:,}")
    print(f"Tokens    : {token_count:,}")
    print(f"Output    : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
