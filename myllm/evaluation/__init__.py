"""
myllm.evaluation — Comprehensive production evaluation subsystem for Dhruva.
"""

from myllm.evaluation.eval_harness import (
    ProductionEvaluationHarness,
    EvaluationSummary,
    EvaluationSampleResult,
    ForceFastRouter,
)
from myllm.evaluation.loader import (
    load_real_checkpoint,
    build_production_orchestrator,
)
from myllm.evaluation.datasets.benchmark_v1 import (
    get_benchmark_dataset,
    BENCHMARK_ITEMS,
)

__all__ = [
    "ProductionEvaluationHarness",
    "EvaluationSummary",
    "EvaluationSampleResult",
    "ForceFastRouter",
    "load_real_checkpoint",
    "build_production_orchestrator",
    "get_benchmark_dataset",
    "BENCHMARK_ITEMS",
]
