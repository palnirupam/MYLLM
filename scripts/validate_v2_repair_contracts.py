"""Dependency-free static contract checks for the V2 repair candidate."""

from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent


def require(path: str, needles: list[str]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{path} missing required contract: {needle}")


def main() -> int:
    config = (ROOT / "configs/dhruva_v2_t4.yaml").read_text(encoding="utf-8")
    for key, value in {
        "vocab_size": "48000", "d_model": "768", "n_layers": "12",
        "n_heads": "12", "n_kv_heads": "4", "intermediate_size": "2048",
        "max_seq_len": "1024", "qk_norm": "true", "expected_parameters": "112382208",
    }.items():
        if not re.search(rf"{re.escape(key)}:\s*{re.escape(value)}", config):
            raise AssertionError(f"config invariant failed: {key}={value}")
    require("myllm/training/checkpoint/manager.py", ["REQUIRED_TRAINING_STATE", "COMPLETE", "validate_checkpoint", "rank_rng_states", "prune"])
    require("myllm/training/pretraining/trainer.py", ["cumulative_loss_tokens", "self.model.no_sync()", "all_gather_object", "packed corpus exhausted"])
    require("myllm/training/data/packed.py", ["tokens_sha256", "tokens_bytes", "COMPLETE", "np.memmap"])
    require("scripts/build_v2_packed_corpus.py", ["tempfile.mkdtemp", "os.fsync", "tokens_sha256", "tokens_bytes"])
    require("scripts/run_kaggle_v2.py", ["approved Kaggle execution contract", "expected_parameter_count", "require_training_state=True"])
    require("myllm/utils/runtime_guard.py", ["DHRUVA_KAGGLE_RUNNER", "KAGGLE_KERNEL_RUN_TYPE"])
    ast.parse((ROOT / "scripts/run_kaggle_v2.py").read_text(encoding="utf-8"))
    ast.parse((ROOT / "myllm/training/checkpoint/manager.py").read_text(encoding="utf-8"))
    print("DHRUVA V2 REPAIR CONTRACTS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
