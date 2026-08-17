"""
tests/unit/test_orchestrator.py
Verifies end-to-end orchestration in Phase 1: routing, state management, fast path, and telemetry.
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.schemas import RoutePath, VerificationStatus, Document
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.runtime.interfaces.base import InferenceRuntime


class MockInferenceRuntime(InferenceRuntime):
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        if "Paris is the capital" in prompt:
            return prompt + " Paris is the capital of France."
        return prompt + " Dhruva response generated."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " Dhruva response generated."


def test_orchestrator_fast_path():
    runtime = MockInferenceRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("Hello!")

    assert result.route_taken == RoutePath.FAST
    assert result.verification_status == VerificationStatus.PASS
    assert "Dhruva response generated" in result.answer
    assert result.confidence > 0.5
    assert "latency_ms" in result.telemetry
    assert result.telemetry["route"] == "FAST"
    assert result.telemetry["tokens_consumed"] > 0


def test_orchestrator_abstention():
    runtime = MockInferenceRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("predict exact lottery numbers for next month")

    assert result.route_taken == RoutePath.ABSTAIN
    assert result.verification_status == VerificationStatus.ABSTAIN
    assert "do not have sufficient verifiable information" in result.answer
    assert result.confidence == 0.0
    assert result.uncertainty_reason is not None


def test_orchestrator_tool_path():
    runtime = MockInferenceRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    # A math query routes to TOOL and executes SafeCalculatorTool
    result = orchestrator.answer("Calculate 25 * 40")

    assert result.route_taken == RoutePath.TOOL
    assert result.verification_status == VerificationStatus.PASS
    assert "25 * 40 = 1000" in result.answer
    assert "calculator" in result.tools_used


def test_orchestrator_retrieve_path():
    runtime = MockInferenceRuntime()
    retriever = InMemoryBM25Retriever()
    doc = Document(
        doc_id="paris_doc",
        content="Paris is the capital of France.",
        source="Geography Encyclopedia",
    )
    retriever.add_documents([doc])
    orchestrator = DhruvaOrchestrator(runtime=runtime, retriever=retriever)

    # Query routes to RETRIEVE and verifies with evidence
    result = orchestrator.answer("cite sources for capital of France")

    assert result.route_taken == RoutePath.RETRIEVE
    assert result.verification_status == VerificationStatus.PASS
    assert "paris_doc" in result.evidence_citations


def test_orchestrator_think_path():
    runtime = MockInferenceRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    # A reasoning query routes to THINK and executes bounded ThinkPath
    result = orchestrator.answer("step-by-step derive the formula for kinetic energy")

    assert result.route_taken == RoutePath.THINK
    assert result.verification_status == VerificationStatus.PASS
    assert result.telemetry["route"] == "THINK"


if __name__ == "__main__":
    tests = [
        test_orchestrator_fast_path,
        test_orchestrator_abstention,
        test_orchestrator_tool_path,
        test_orchestrator_retrieve_path,
        test_orchestrator_think_path,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL ORCHESTRATOR TESTS PASSED")
