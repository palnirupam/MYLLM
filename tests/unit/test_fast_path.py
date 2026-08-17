"""
tests/unit/test_fast_path.py
Verifies FastPath execution and token usage accounting.
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.paths.fast import FastPath
from myllm.intelligence.schemas import ExecutionState, RouteDecision, RoutePath, ComputeBudget
from myllm.runtime.interfaces.base import InferenceRuntime


class MockInferenceRuntime(InferenceRuntime):
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        return prompt + " Hello! I am Dhruva, a fast and factual assistant."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " Hello!"


def test_fast_path_execution():
    runtime = MockInferenceRuntime()
    fast_path = FastPath()

    budget = ComputeBudget(max_tokens=64)
    decision = RouteDecision(path=RoutePath.FAST, routing_score=0.8, reason="Fast greeting", budget=budget)
    state = ExecutionState(query="Hi", route_decision=decision, budget=budget)

    output = fast_path.execute("Hi", state, runtime)

    assert "Hello! I am Dhruva" in output.text
    assert output.tokens_generated > 0
    assert state.tokens_consumed == output.tokens_generated
    assert output.confidence_estimate == 0.85


if __name__ == "__main__":
    test_fast_path_execution()
    print("  PASS  test_fast_path_execution")
    print("\nALL FAST PATH TESTS PASSED")
