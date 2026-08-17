"""
myllm.intelligence.reasoning.benchmark — A/B comparative benchmark for FAST-only vs Adaptive (FAST + THINK).
Measures accuracy, repetition rate, latency, token consumption, and compute proxy.
"""

from typing import List, Dict, Any, Optional
import time
from myllm.intelligence.schemas import RoutePath, VerificationStatus
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.router.base import BaseRouter
from myllm.intelligence.router.budget import BudgetAllocator
from myllm.intelligence.schemas import RouteDecision
from myllm.runtime.interfaces.base import InferenceRuntime
from myllm.intelligence.verifier.structural import analyze_repetition


class FastOnlyRouter(BaseRouter):
    """Router that forces all queries to FAST path unconditionally."""
    def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        return RouteDecision(
            path=RoutePath.FAST,
            routing_score=1.0,
            reason="Forced FastOnly baseline",
            budget=BudgetAllocator.allocate(RoutePath.FAST),
        )


BENCHMARK_REASONING_PROMPTS = [
    "Step-by-step derive the conservation of linear momentum from Newton's laws.",
    "Solve the logic puzzle: Three boxes are labeled incorrectly. Box A has apples, Box B has oranges, Box C has mixed. You draw one fruit from Box C. How do you label all boxes?",
    "Compare and contrast in detail the time complexity and auxiliary space of QuickSort vs MergeSort.",
    "Explain step-by-step how a compiler converts high-level source code into machine code.",
]


def run_reasoning_ab_benchmark(runtime: InferenceRuntime) -> Dict[str, Any]:
    """
    Runs synthetic comparative unit evaluation between:
    Variant A: FAST-only (Single pass, no escalation)
    Variant B: Adaptive Reasoning (RuleRouter + ThinkPath + Escalation)

    NOTE: This is an infrastructure & verifier test benchmark.
    It does not measure real pre-trained weights capability without a trained checkpoint.
    """
    orch_a = DhruvaOrchestrator(runtime=runtime, router=FastOnlyRouter())
    orch_b = DhruvaOrchestrator(runtime=runtime)  # Default adaptive router

    results_a = []
    results_b = []

    for prompt in BENCHMARK_REASONING_PROMPTS:
        # Run Variant A
        start_a = time.time()
        res_a = orch_a.answer(prompt)
        lat_a = (time.time() - start_a) * 1000.0
        rep_a = analyze_repetition(res_a.answer)

        results_a.append({
            "prompt": prompt,
            "route": res_a.route_taken.value,
            "status": res_a.verification_status.value,
            "confidence": res_a.confidence,
            "repetition_score": round(rep_a, 3),
            "latency_ms": round(lat_a, 2),
            "tokens": res_a.telemetry.get("tokens_consumed", 0),
        })

        # Run Variant B
        start_b = time.time()
        res_b = orch_b.answer(prompt)
        lat_b = (time.time() - start_b) * 1000.0
        rep_b = analyze_repetition(res_b.answer)

        results_b.append({
            "prompt": prompt,
            "route": res_b.route_taken.value,
            "status": res_b.verification_status.value,
            "confidence": res_b.confidence,
            "repetition_score": round(rep_b, 3),
            "latency_ms": round(lat_b, 2),
            "tokens": res_b.telemetry.get("tokens_consumed", 0),
        })

    # Aggregates
    a_pass_rate = sum(1 for r in results_a if r["status"] == "PASS") / len(results_a)
    a_avg_rep = sum(r["repetition_score"] for r in results_a) / len(results_a)
    a_avg_lat = sum(r["latency_ms"] for r in results_a) / len(results_a)
    a_avg_tokens = sum(r["tokens"] for r in results_a) / len(results_a)

    b_pass_rate = sum(1 for r in results_b if r["status"] == "PASS") / len(results_b)
    b_avg_rep = sum(r["repetition_score"] for r in results_b) / len(results_b)
    b_avg_lat = sum(r["latency_ms"] for r in results_b) / len(results_b)
    b_avg_tokens = sum(r["tokens"] for r in results_b) / len(results_b)

    return {
        "variant_a_fast_only": {
            "verified_pass_rate": round(a_pass_rate, 2),
            "average_repetition": round(a_avg_rep, 3),
            "average_latency_ms": round(a_avg_lat, 2),
            "generated_tokens": round(a_avg_tokens, 1),
            "compute_proxy": round(a_avg_tokens * 1.0, 1),
            "results": results_a,
        },
        "variant_b_adaptive_think": {
            "verified_pass_rate": round(b_pass_rate, 2),
            "average_repetition": round(b_avg_rep, 3),
            "average_latency_ms": round(b_avg_lat, 2),
            "generated_tokens": round(b_avg_tokens, 1),
            "compute_proxy": round(b_avg_tokens * 1.0, 1),
            "results": results_b,
        },
    }
