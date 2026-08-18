#!/usr/bin/env python3
"""Fail-closed quality and contamination gate for a Dhruva V2 JSONL corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from myllm.data.quality import QualityPolicy, analyze_document, flatten_text_values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_eval_fragments(path: Path | None) -> set[str]:
    fragments: set[str] = set()
    if path is None:
        return fragments
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid eval JSON at line {line_number}: {exc}") from exc
            for value in flatten_text_values(record):
                folded = value.casefold()
                if len(folded) >= 80:
                    fragments.add(folded)
    return fragments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Candidate pretraining JSONL")
    parser.add_argument("--eval", required=True, help="Held-out validation/benchmark JSONL")
    parser.add_argument("--manifest", default="artifacts/v2_corpus_quality_manifest.json")
    parser.add_argument("--cleaned-corpus", help="Optional normalized accepted JSONL output")
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-reject-rate", type=float, default=0.02)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.005)
    parser.add_argument("--max-script-warning-rate", type=float, default=0.0)
    parser.add_argument("--allow-missing-metadata", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    eval_path = Path(args.eval) if args.eval else None
    manifest_path = Path(args.manifest)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    if eval_path is not None and not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    policy = QualityPolicy(
        min_chars=args.min_chars,
        require_metadata=not args.allow_missing_metadata,
    )
    eval_fragments = load_eval_fragments(eval_path)

    totals = Counter()
    reasons = Counter()
    warnings = Counter()
    languages = Counter()
    domains = Counter()
    sources = Counter()
    scripts = Counter()
    seen_hashes: set[str] = set()
    contaminated_lines: list[int] = []
    invalid_json_lines: list[int] = []
    cleaned_path = Path(args.cleaned_corpus) if args.cleaned_corpus else None
    temp_cleaned_path = None
    if cleaned_path:
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{cleaned_path.name}.", dir=str(cleaned_path.parent), text=True)
        os.close(fd)
        temp_cleaned_path = Path(temp_name)
    cleaned_handle = temp_cleaned_path.open("w", encoding="utf-8") if temp_cleaned_path else None

    try:
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                totals["documents"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_lines.append(line_number)
                    reasons["invalid_json"] += 1
                    totals["rejected"] += 1
                    continue

                if not isinstance(record, dict):
                    reasons["record_not_object"] += 1
                    totals["rejected"] += 1
                    continue

                result = analyze_document(record, policy)
                if result.content_hash in seen_hashes:
                    reasons["normalized_exact_duplicate"] += 1
                    totals["duplicates"] += 1
                    totals["rejected"] += 1
                    continue
                seen_hashes.add(result.content_hash)

                contaminated = any(fragment in result.normalized_text.casefold() for fragment in eval_fragments)
                if contaminated:
                    contaminated_lines.append(line_number)
                    reasons["heldout_contamination"] += 1
                    totals["rejected"] += 1
                    continue

                if not result.accepted:
                    reasons.update(result.reasons)
                    totals["rejected"] += 1
                    continue

                totals["accepted"] += 1
                totals["accepted_characters"] += len(result.normalized_text)
                warnings.update(result.warnings)
                scripts[result.dominant_script] += 1
                languages[str(record.get("language", "Unknown"))] += 1
                domains[str(record.get("domain", "Unknown"))] += 1
                sources[str(record.get("source", "Unknown"))] += 1
                if cleaned_handle:
                    cleaned_record = dict(record)
                    cleaned_record["text"] = result.normalized_text
                    cleaned_handle.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
    finally:
        if cleaned_handle:
            cleaned_handle.close()
    documents = max(1, totals["documents"])
    reject_rate = totals["rejected"] / documents
    duplicate_rate = totals["duplicates"] / documents
    script_warning_count = sum(count for key, count in reasons.items() if key.startswith("language_script_mismatch:"))
    script_warning_rate = script_warning_count / max(1, totals["documents"])

    failures: list[str] = []
    if invalid_json_lines:
        failures.append(f"invalid_json_lines={len(invalid_json_lines)}")
    if contaminated_lines:
        failures.append(f"heldout_contamination={len(contaminated_lines)}")
    if reject_rate > args.max_reject_rate:
        failures.append(f"reject_rate={reject_rate:.4%}>{args.max_reject_rate:.4%}")
    if duplicate_rate > args.max_duplicate_rate:
        failures.append(f"duplicate_rate={duplicate_rate:.4%}>{args.max_duplicate_rate:.4%}")
    if script_warning_rate > args.max_script_warning_rate:
        failures.append(
            f"script_warning_rate={script_warning_rate:.4%}>{args.max_script_warning_rate:.4%}"
        )
    if totals["accepted"] == 0:
        failures.append("no_accepted_documents")

    if cleaned_path and temp_cleaned_path and not failures:
        temp_cleaned_path.replace(cleaned_path)
    elif temp_cleaned_path:
        temp_cleaned_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "status": "PASS" if not failures else "FAIL",
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_bytes": corpus_path.stat().st_size,
        "cleaned_corpus": str(cleaned_path) if cleaned_path else None,
        "cleaned_corpus_sha256": sha256_file(cleaned_path) if cleaned_path and cleaned_path.exists() else None,
        "cleaned_corpus_bytes": cleaned_path.stat().st_size if cleaned_path and cleaned_path.exists() else None,
        "eval": str(eval_path) if eval_path else None,
        "eval_sha256": sha256_file(eval_path) if eval_path else None,
        "policy": {
            "min_chars": policy.min_chars,
            "max_chars": policy.max_chars,
            "require_metadata": policy.require_metadata,
            "max_reject_rate": args.max_reject_rate,
            "max_duplicate_rate": args.max_duplicate_rate,
            "max_script_warning_rate": args.max_script_warning_rate,
        },
        "counts": dict(totals),
        "rates": {
            "reject": reject_rate,
            "duplicate": duplicate_rate,
            "script_warning": script_warning_rate,
        },
        "rejection_reasons": dict(reasons),
        "warnings": dict(warnings),
        "preprocessing_version": "v2-quality-1",
        "required_metadata": list(policy.required_metadata),
        "language_documents": dict(languages),
        "domain_documents": dict(domains),
        "source_documents": dict(sources),
        "dominant_script_documents": dict(scripts),
        "invalid_json_lines": invalid_json_lines[:100],
        "contaminated_lines": contaminated_lines[:100],
        "failures": failures,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"DHRUVA V2 CORPUS GATE: {manifest['status']}")
    print(f"Documents  : {totals['documents']:,}")
    print(f"Accepted   : {totals['accepted']:,}")
    print(f"Rejected   : {totals['rejected']:,} ({reject_rate:.3%})")
    print(f"Duplicates : {totals['duplicates']:,} ({duplicate_rate:.3%})")
    print(f"Manifest   : {manifest_path}")
    for failure in failures:
        print(f"FAILURE    : {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
