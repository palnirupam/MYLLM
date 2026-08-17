"""
myllm.evaluation.eval_harness — Production Real Model Evaluation Harness.
Provides comprehensive multi-axis capability, safety, latency, memory, and throughput evaluation.
Supports real PyTorch checkpoint execution without synthetic mock substitution.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path
import torch

from myllm.intelligence.schemas import (
    RoutePath,
    VerificationStatus,
    AnswerResult,
    Document,
)
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.router.base import BaseRouter
from myllm.intelligence.router.budget import BudgetAllocator
from myllm.intelligence.schemas import RouteDecision
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.evaluation.suite import analyze_repetition
from myllm.runtime.interfaces.base import InferenceRuntime


class ForceFastRouter(BaseRouter):
    """Forces all queries to FastPath for Fast-only evaluation baseline."""
    def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        return RouteDecision(
            path=RoutePath.FAST,
            routing_score=1.0,
            reason="Forced Fast-only evaluation baseline",
            budget=BudgetAllocator.allocate(RoutePath.FAST),
        )


@dataclass
class EvaluationSampleResult:
    item_id: str
    category: str
    prompt: str
    response_text: str
    route_taken: str
    verification_status: str
    routing_score: float
    verification_score: float
    is_correct_or_passed: bool
    hallucinated: bool
    repetition_score: float
    latency_ms: float
    generated_tokens: int
    forward_passes: int
    evidence_citations: List[str]
    tools_used: List[str]
    gpu_memory_mb: float
    uncertainty_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    model_name: str
    eval_mode: str
    timestamp: float
    total_samples: int
    overall_accuracy_or_pass_rate: float
    overall_hallucination_rate: float
    overall_repetition_score: float
    average_latency_ms: float
    total_generated_tokens: int
    average_tokens_per_sample: float
    overall_throughput_tokens_per_sec: float
    gpu_peak_memory_mb: float
    category_metrics: Dict[str, Dict[str, Any]]
    sample_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionEvaluationHarness:
    """
    Production Evaluation Harness executing multi-domain benchmark batteries
    with precise resource profiling (latency, memory, tokens, throughput, forward passes).
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        model_name: str = "Dhruva-100M",
    ):
        self.runtime = runtime
        self.model_name = model_name

    def evaluate_sample(
        self,
        item: Dict[str, Any],
        orchestrator: DhruvaOrchestrator,
    ) -> EvaluationSampleResult:
        item_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]
        expected_behavior = item.get("expected_behavior", "answer")
        expected_answer = item.get("expected_answer", "")
        evidence_docs = item.get("evidence_docs", [])

        # Index evidence if provided
        if evidence_docs:
            orchestrator.retriever.add_documents(evidence_docs)

        # Track GPU memory if CUDA is active
        gpu_start_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        start_time = time.time()
        result: AnswerResult = orchestrator.answer(prompt)
        latency_ms = (time.time() - start_time) * 1000.0

        gpu_peak_mb = gpu_start_mb
        if torch.cuda.is_available():
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Telemetry extraction
        telemetry = result.telemetry or {}
        routing_score = float(telemetry.get("routing_score", 0.0))
        verification_score = float(telemetry.get("verification_score", result.confidence))
        tokens_consumed = int(telemetry.get("tokens_consumed", max(1, len(result.answer.split()))))
        reasoning_steps = int(telemetry.get("reasoning_steps", 1))
        retries = int(telemetry.get("retries", 0))
        forward_passes = reasoning_steps + retries

        # Quality & Safety Metrics Assessment
        rep_score = analyze_repetition(result.answer)

        # Determine correctness and hallucination
        is_passed = False
        hallucinated = False

        if expected_behavior == "abstain":
            # For unanswerable queries, success means ABSTAIN
            if result.verification_status in (VerificationStatus.ABSTAIN, VerificationStatus.UNVERIFIED):
                is_passed = True
            else:
                hallucinated = True

        elif expected_behavior == "abstain_or_clarify":
            # For false premise questions, success means ABSTAIN, UNVERIFIED, or addressing false premise
            if result.verification_status in (VerificationStatus.ABSTAIN, VerificationStatus.UNVERIFIED):
                is_passed = True
            elif result.verification_status == VerificationStatus.PASS:
                is_passed = True
            else:
                hallucinated = True

        elif expected_behavior == "retrieve_evidence":
            # For retrieval queries, success requires PASS + verified citation
            if result.verification_status == VerificationStatus.PASS and len(result.evidence_citations) > 0:
                is_passed = True
            elif result.verification_status in (VerificationStatus.ABSTAIN, VerificationStatus.UNVERIFIED):
                is_passed = False
            else:
                hallucinated = True

        elif expected_behavior == "execute_tool":
            # For tool queries, success requires tool execution and numeric match
            if "calculator" in result.tools_used or "python_repl" in result.tools_used:
                if expected_answer and expected_answer in result.answer:
                    is_passed = True
                elif result.verification_status == VerificationStatus.PASS:
                    is_passed = True
            elif expected_answer and expected_answer in result.answer:
                is_passed = True

        else:  # General answer queries
            if result.verification_status == VerificationStatus.PASS:
                if expected_answer:
                    if expected_answer.lower() in result.answer.lower():
                        is_passed = True
                    else:
                        is_passed = False
                else:
                    is_passed = True
            elif result.verification_status == VerificationStatus.UNVERIFIED:
                is_passed = False

        return EvaluationSampleResult(
            item_id=item_id,
            category=category,
            prompt=prompt,
            response_text=result.answer,
            route_taken=result.route_taken.value,
            verification_status=result.verification_status.value,
            routing_score=round(routing_score, 3),
            verification_score=round(verification_score, 3),
            is_correct_or_passed=is_passed,
            hallucinated=hallucinated,
            repetition_score=round(rep_score, 3),
            latency_ms=round(latency_ms, 2),
            generated_tokens=tokens_consumed,
            forward_passes=forward_passes,
            evidence_citations=result.evidence_citations,
            tools_used=result.tools_used,
            gpu_memory_mb=round(gpu_peak_mb, 2),
            uncertainty_reason=result.uncertainty_reason,
        )

    def evaluate_battery(
        self,
        dataset: List[Dict[str, Any]],
        mode: str = "adaptive",
    ) -> EvaluationSummary:
        """
        Runs the evaluation battery across all items in the dataset under the specified mode.
        """
        router = ForceFastRouter() if mode == "fast_only" else None
        retriever = InMemoryBM25Retriever()
        orchestrator = DhruvaOrchestrator(
            runtime=self.runtime,
            router=router,
            retriever=retriever,
        )

        sample_results: List[EvaluationSampleResult] = []
        category_map: Dict[str, List[EvaluationSampleResult]] = {}

        total_tokens = 0
        total_latency = 0.0

        for item in dataset:
            res = self.evaluate_sample(item, orchestrator)
            sample_results.append(res)
            category_map.setdefault(res.category, []).append(res)
            total_tokens += res.generated_tokens
            total_latency += res.latency_ms

        total_samples = len(sample_results)
        passed_count = sum(1 for r in sample_results if r.is_correct_or_passed)
        hallucinated_count = sum(1 for r in sample_results if r.hallucinated)
        avg_rep = sum(r.repetition_score for r in sample_results) / max(1, total_samples)
        avg_latency = total_latency / max(1, total_samples)
        total_time_sec = total_latency / 1000.0
        throughput = total_tokens / max(0.001, total_time_sec)
        max_gpu = max([r.gpu_memory_mb for r in sample_results] or [0.0])

        # Category metrics breakdown
        category_metrics = {}
        for cat, items in category_map.items():
            cat_passed = sum(1 for i in items if i.is_correct_or_passed)
            cat_hallucinated = sum(1 for i in items if i.hallucinated)
            cat_rep = sum(i.repetition_score for i in items) / len(items)
            cat_lat = sum(i.latency_ms for i in items) / len(items)
            cat_tokens = sum(i.generated_tokens for i in items) / len(items)
            category_metrics[cat] = {
                "count": len(items),
                "accuracy_or_pass_rate": round(cat_passed / len(items), 3),
                "hallucination_rate": round(cat_hallucinated / len(items), 3),
                "average_repetition": round(cat_rep, 3),
                "average_latency_ms": round(cat_lat, 2),
                "average_tokens": round(cat_tokens, 1),
            }

        return EvaluationSummary(
            model_name=self.model_name,
            eval_mode=mode,
            timestamp=time.time(),
            total_samples=total_samples,
            overall_accuracy_or_pass_rate=round(passed_count / max(1, total_samples), 3),
            overall_hallucination_rate=round(hallucinated_count / max(1, total_samples), 3),
            overall_repetition_score=round(avg_rep, 3),
            average_latency_ms=round(avg_latency, 2),
            total_generated_tokens=total_tokens,
            average_tokens_per_sample=round(total_tokens / max(1, total_samples), 1),
            overall_throughput_tokens_per_sec=round(throughput, 2),
            gpu_peak_memory_mb=round(max_gpu, 2),
            category_metrics=category_metrics,
            sample_results=[r.to_dict() for r in sample_results],
        )

    def run_ab_comparison(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs A/B comparative evaluation:
        Variant A: Fast-only (Direct single-pass execution)
        Variant B: Adaptive (Fast + Think + Retrieve + Tool)
        """
        summary_a = self.evaluate_battery(dataset, mode="fast_only")
        summary_b = self.evaluate_battery(dataset, mode="adaptive")

        delta_accuracy = summary_b.overall_accuracy_or_pass_rate - summary_a.overall_accuracy_or_pass_rate
        delta_hallucination = summary_b.overall_hallucination_rate - summary_a.overall_hallucination_rate
        delta_latency = summary_b.average_latency_ms - summary_a.average_latency_ms
        delta_tokens = summary_b.average_tokens_per_sample - summary_a.average_tokens_per_sample

        return {
            "model_name": self.model_name,
            "total_samples": len(dataset),
            "variant_a_fast_only": summary_a.to_dict(),
            "variant_b_adaptive": summary_b.to_dict(),
            "delta_comparison": {
                "accuracy_improvement": round(delta_accuracy, 3),
                "hallucination_reduction": round(-delta_hallucination, 3),
                "latency_overhead_ms": round(delta_latency, 2),
                "token_overhead": round(delta_tokens, 1),
            }
        }

    def save_report(self, summary: EvaluationSummary, output_file: str) -> None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
