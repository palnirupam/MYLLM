"""Guard local entry points against accidental CUDA training."""

from __future__ import annotations

import os


def assert_training_environment() -> None:
    approved = os.environ.get("DHRUVA_KAGGLE_RUNNER") == "1" or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    if not approved:
        raise RuntimeError(
            "Training is disabled outside the approved Kaggle contract. "
            "Use scripts/run_kaggle_v2.py under torchrun on Kaggle."
        )
