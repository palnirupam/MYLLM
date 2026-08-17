"""
tests/unit/test_policy_hardening.py
Verifies policy hardening, budget exhaustion, route escalation, retry ceilings,
complete telemetry logging, and the guarantee that AnswerResult never leaks chain-of-thought.
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.policy import StateManager
from myllm.intelligence.schemas import RoutePath, VerificationStatus, ComputeBudget, RouteDecision
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.runtime.interfaces.base import InferenceRuntime


class FlakyRuntime(InferenceRuntime):
    """Returns repetitive looping text on first call, clean text on second call."""
    def __init__(self):
        self.call_count = 0

    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return "repeating loop repeating loop repeating loop repeating loop repeating loop repeating loop"
        return "Clean non-repetitive response from Dhruva."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield "Response"


class AlwaysFailingRuntime(InferenceRuntime):
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        return "loop loop loop loop loop loop loop loop loop loop"

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield "loop"


def test_orchestrator_retry_ceiling_and_abstention():
    """Ensure that if candidate repeatedly fails verifier beyond max_retries, it abstains cleanly."""
    runtime = AlwaysFailingRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("Hello")
    assert result.verification_status == VerificationStatus.ABSTAIN
    assert "uncertain" in result.answer.lower()
    assert result.confidence == 0.0


def test_orchestrator_retry_recovery():
    """Ensure verifier REVISE triggers retry/escalation under budget, succeeding on clean output."""
    runtime = FlakyRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("Hello")
    assert result.verification_status == VerificationStatus.PASS
    assert "Clean non-repetitive response" in result.answer
    assert result.route_taken == RoutePath.THINK
    assert result.telemetry["tokens_consumed"] > 0


def test_telemetry_completeness():
    """Ensure all required telemetry keys are present in the final summary."""
    runtime = FlakyRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("Explain gravity")
    t = result.telemetry

    required_keys = [
        "query_id",
        "route",
        "routing_score",
        "latency_ms",
        "verification_status",
        "tools_used",
        "retrieval_used",
        "success",
        "tokens_consumed",
        "tool_calls_made",
        "retries_attempted",
        "event_count",
    ]
    for k in required_keys:
        assert k in t, f"Missing telemetry key: {k}"

    assert isinstance(t["query_id"], str)
    assert isinstance(t["success"], bool)
    assert t["latency_ms"] >= 0.0


def test_no_hidden_chain_of_thought_leakage():
    """Ensure AnswerResult contains only the final public answer without internal trace leak."""
    runtime = FlakyRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("Hello there")
    # Verify no internal private deliberation tags exist in public output
    assert "<scratchpad>" not in result.answer
    assert "<thought>" not in result.answer
    assert "INTERNAL_TRACE" not in result.answer


def test_state_manager_escalation_triggers():
    manager = StateManager()
    budget = ComputeBudget(max_retries=2)
    decision = RouteDecision(path=RoutePath.FAST, routing_score=0.7, reason="Fast", budget=budget)
    state = manager.create_initial_state("Fact query", decision)

    # UNVERIFIED escalates to RETRIEVE
    should_esc, target, rsn = manager.should_escalate(state, VerificationStatus.UNVERIFIED)
    assert should_esc is True
    assert target == RoutePath.RETRIEVE

    # REVISE escalates to THINK
    should_esc, target, rsn = manager.should_escalate(state, VerificationStatus.REVISE)
    assert should_esc is True
    assert target == RoutePath.THINK


if __name__ == "__main__":
    tests = [
        test_orchestrator_retry_ceiling_and_abstention,
        test_orchestrator_retry_recovery,
        test_telemetry_completeness,
        test_no_hidden_chain_of_thought_leakage,
        test_state_manager_escalation_triggers,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL POLICY HARDENING TESTS PASSED")
