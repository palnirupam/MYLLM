"""
tests/unit/test_phase4_reasoning.py
Comprehensive Phase 4 Test Suite:
- Bounded ThinkPath execution
- Reasoning budget exhaustion & timeout
- At most one retry / revision attempt
- Verifier critique feedback and revision
- FAST -> THINK escalation
- Entity-attribution check in EvidenceGroundedVerifier
- Scratchpad / private chain-of-thought privacy
- A/B reasoning benchmark harness
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.paths.think import ThinkPath
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.schemas import (
    ExecutionState,
    RouteDecision,
    RoutePath,
    VerificationStatus,
    ComputeBudget,
    Document,
)
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.verifier.factual import EvidenceGroundedVerifier
from myllm.intelligence.reasoning.benchmark import run_reasoning_ab_benchmark
from myllm.runtime.interfaces.base import InferenceRuntime


class MockReasoningRuntime(InferenceRuntime):
    """Runtime that produces high quality verified reasoning on math / logic."""
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        if "Correction Note" in prompt:
            return "Step 1: Apply law. Step 2: Integrate force over time. The momentum is strictly conserved."
        if "derive" in prompt.lower() or "step-by-step" in prompt.lower():
            return "Step 1: Newton's third law states F_12 = -F_21. Step 2: dP/dt = 0, so total momentum P is constant."
        return prompt + " Synthesized reasoning output."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " Step 1"


class FailingReasoningRuntime(InferenceRuntime):
    """Generates repetitive or flawed text that fails verifier even after revision."""
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        return "<scratchpad> private thought </scratchpad> loop loop loop loop loop loop loop"

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " loop"


# 1. Bounded ThinkPath Execution
def test_think_path_bounded_execution():
    runtime = MockReasoningRuntime()
    think_path = ThinkPath()

    budget = ComputeBudget(max_tokens=128, max_reasoning_steps=2)
    decision = RouteDecision(path=RoutePath.THINK, routing_score=0.85, reason="Deep logic", budget=budget)
    state = ExecutionState(query="Derive momentum conservation", route_decision=decision, budget=budget)

    output = think_path.execute("Derive momentum conservation", state, runtime)

    assert "Newton's third law" in output.text
    assert state.reasoning_steps_taken <= 2
    assert "<scratchpad>" not in output.text


# 2. Maximum One Revision Attempt
def test_think_path_max_one_retry_ceiling():
    runtime = FailingReasoningRuntime()
    think_path = ThinkPath()

    budget = ComputeBudget(max_tokens=128, max_reasoning_steps=3, max_retries=1)
    decision = RouteDecision(path=RoutePath.THINK, routing_score=0.85, reason="Deep logic", budget=budget)
    state = ExecutionState(query="Flawed prompt", route_decision=decision, budget=budget)

    output = think_path.execute("Flawed prompt", state, runtime)

    # Exactly 1 initial step + at most 1 revision step = max 2 steps taken
    assert state.reasoning_steps_taken <= 2
    assert state.retries_attempted <= 1


# 3. Hidden Scratchpad Privacy Guarantee
def test_hidden_scratchpad_never_in_answer_result():
    runtime = FailingReasoningRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    result = orchestrator.answer("step-by-step logic derivation")

    # Output should not leak internal tags
    assert "<scratchpad>" not in result.answer
    assert "</scratchpad>" not in result.answer


# 4. FAST -> THINK Escalation on Verification Failure
def test_fast_to_think_escalation():
    class EscalatingRuntime(InferenceRuntime):
        def __init__(self):
            self.calls = 0
        def load_model(self, model_path: str) -> None:
            pass
        def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
            self.calls += 1
            if self.calls == 1:
                # Fast pass produces repetitive text
                return "repeat repeat repeat repeat repeat repeat repeat repeat"
            # Think pass produces clean structured reasoning
            return "Here is the clear, structured explanation with complete details."
        def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
            yield " Token"

    runtime = EscalatingRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    # Prompt routes to FAST initially
    result = orchestrator.answer("Explain the core ideas of thermodynamics")

    assert result.route_taken == RoutePath.THINK  # Successfully escalated from FAST to THINK
    assert result.verification_status == VerificationStatus.PASS
    assert "structured explanation" in result.answer


# 5. Entity-Attribution Check in EvidenceGroundedVerifier (Phase 3 Correction)
def test_entity_attribution_check():
    verifier = EvidenceGroundedVerifier()

    # Query asks about 'Galileo Galilei'
    query = "Who was Galileo Galilei and what did he observe?"

    # Evidence is topically about astronomy/telescopes, but does NOT mention Galileo
    generic_evidence = ["[doc_id='ast_01'] The telescope was used to observe moons orbiting Jupiter and the rings of Saturn."]
    ans = "Galileo observed moons orbiting Jupiter using an early telescope."

    # Must FAIL entity attribution and return REVISE
    res = verifier.verify(query, ans, evidence=generic_evidence)
    assert res.status == VerificationStatus.REVISE
    assert res.details["entity_attribution_passed"] is False
    assert "Entity attribution failure" in res.critique

    # Now provide evidence containing 'Galileo'
    attributed_evidence = ["[doc_id='ast_02'] Galileo Galilei observed four large moons of Jupiter using his telescope in 1610."]
    res_pass = verifier.verify(query, ans, evidence=attributed_evidence)
    assert res_pass.status == VerificationStatus.PASS
    assert res_pass.details["entity_attribution_passed"] is True


# 6. End-to-End ThinkPath Execution in Orchestrator
def test_orchestrator_think_path_e2e():
    runtime = MockReasoningRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    query = "Step-by-step prove that momentum is conserved in a closed system."
    result = orchestrator.answer(query)

    assert result.route_taken == RoutePath.THINK
    assert result.verification_status == VerificationStatus.PASS
    assert "Newton's third law" in result.answer
    assert result.telemetry["route"] == "THINK"


# 7. A/B Reasoning Benchmark Execution
def test_reasoning_ab_benchmark_harness():
    runtime = MockReasoningRuntime()
    bench = run_reasoning_ab_benchmark(runtime)

    assert "variant_a_fast_only" in bench
    assert "variant_b_adaptive_think" in bench

    var_a = bench["variant_a_fast_only"]
    var_b = bench["variant_b_adaptive_think"]

    assert var_b["verified_pass_rate"] >= var_a["verified_pass_rate"]
    assert var_b["compute_proxy"] >= 0.0


if __name__ == "__main__":
    tests = [
        test_think_path_bounded_execution,
        test_think_path_max_one_retry_ceiling,
        test_hidden_scratchpad_never_in_answer_result,
        test_fast_to_think_escalation,
        test_entity_attribution_check,
        test_orchestrator_think_path_e2e,
        test_reasoning_ab_benchmark_harness,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL PHASE 4 REASONING TESTS PASSED")
