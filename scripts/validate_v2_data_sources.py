#!/usr/bin/env python3
"""Validate the immutable Dhruva V2 Hugging Face source catalog.

This command does not download data. It validates the contract that a later
streaming builder must enforce before any source is admitted to the corpus.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


SHA256_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FIELDS = {
    "id", "dataset", "config", "split", "revision", "text_column",
    "language", "domain", "target_token_share", "license",
    "license_url", "allowed_for_base_pretraining",
}


def load_plan(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data source plan must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported data source plan schema")
    if payload.get("source_revision_policy") != "exact_huggingface_commit":
        raise ValueError("source revisions must be immutable Hugging Face commits")
    if payload.get("token_share_basis") != "final_frozen_tokenizer_tokens":
        raise ValueError("token shares must be measured after final tokenization")
    if payload.get("generated_text_allowed") is not False:
        raise ValueError("generated text must be disabled for base pretraining")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source plan must contain at least one source")
    return payload


def validate_plan(payload: dict, require_complete: bool = True) -> list[str]:
    failures: list[str] = []
    ids: set[str] = set()
    total_share = 0.0
    allowed_share = 0.0
    blocked: list[str] = []

    for index, source in enumerate(payload["sources"]):
        prefix = f"sources[{index}]"
        if not isinstance(source, dict):
            failures.append(f"{prefix} must be a mapping")
            continue
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            failures.append(f"{prefix} missing fields: {sorted(missing)}")

        source_id = str(source.get("id", ""))
        if not source_id or source_id in ids:
            failures.append(f"{prefix} has a missing or duplicate id: {source_id!r}")
        ids.add(source_id)

        revision = str(source.get("revision", ""))
        if not SHA256_RE.fullmatch(revision):
            failures.append(f"{source_id}: revision must be a 40-character lowercase commit")

        share = source.get("target_token_share")
        if not isinstance(share, (int, float)) or not 0.0 <= float(share) <= 1.0:
            failures.append(f"{source_id}: target_token_share must be within [0, 1]")
        else:
            total_share += float(share)

        allowed = source.get("allowed_for_base_pretraining")
        if not isinstance(allowed, bool):
            failures.append(f"{source_id}: allowed_for_base_pretraining must be boolean")
        elif allowed:
            allowed_share += float(share or 0.0)
        else:
            blocked.append(source_id)
            if not source.get("blocking_reason"):
                failures.append(f"{source_id}: blocked sources require blocking_reason")

        for field in ("dataset", "config", "split", "text_column", "language", "domain", "license", "license_url"):
            if not str(source.get(field, "")).strip():
                failures.append(f"{source_id}: {field} must not be empty")

    if abs(total_share - 1.0) > 1e-9:
        failures.append(f"target token shares sum to {total_share:.12f}, expected 1.0")
    if require_complete and blocked:
        failures.append(
            "base-pretraining plan is incomplete; blocked sources: " + ", ".join(blocked)
        )
    if allowed_share > 1.0 + 1e-9:
        failures.append(f"allowed token shares exceed 1.0: {allowed_share:.12f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/dhruva_v2_data_sources.yaml")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="validate the catalog while allowing explicitly blocked sources",
    )
    args = parser.parse_args()

    path = Path(args.plan)
    payload = load_plan(path)
    failures = validate_plan(payload, require_complete=not args.catalog_only)
    sources = payload["sources"]
    total = sum(float(item["target_token_share"]) for item in sources)
    allowed = sum(
        float(item["target_token_share"])
        for item in sources
        if item.get("allowed_for_base_pretraining") is True
    )
    blocked = [item["id"] for item in sources if not item.get("allowed_for_base_pretraining", False)]

    print(f"DHRUVA V2 DATA SOURCE PLAN: {'PASS' if not failures else 'FAIL'}")
    print(f"Sources             : {len(sources)}")
    print(f"Planned token share : {total:.3f}")
    print(f"Allowed token share : {allowed:.3f}")
    print(f"Blocked sources     : {len(blocked)}")
    if blocked:
        print("Blocked             : " + ", ".join(blocked))
    for failure in failures:
        print(f"FAILURE             : {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
