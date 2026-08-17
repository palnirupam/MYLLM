"""
tests/unit/test_router.py
Verifies deterministic rule routing, score bounds, and compute budget allocations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.router.rules import RuleRouter
from myllm.intelligence.router.budget import BudgetAllocator
from myllm.intelligence.schemas import RoutePath


def test_rule_router_math_calc():
    router = RuleRouter()
    decision = router.route("What is 154 + 892?")
    assert decision.path == RoutePath.TOOL
    assert 0.0 <= decision.routing_score <= 1.0
    assert "math_calc_pattern" in decision.metadata["matched_rules"]


def test_rule_router_code():
    router = RuleRouter()
    decision = router.route("Write a Python function to compute fibonacci numbers.")
    assert decision.path == RoutePath.TOOL
    assert 0.0 <= decision.routing_score <= 1.0
    assert "code_execution_pattern" in decision.metadata["matched_rules"]


def test_rule_router_think():
    router = RuleRouter()
    decision = router.route("Please explain step-by-step how general relativity explains gravity.")
    assert decision.path == RoutePath.THINK
    assert 0.0 <= decision.routing_score <= 1.0
    assert "deep_think_pattern" in decision.metadata["matched_rules"]


def test_rule_router_retrieval():
    router = RuleRouter()
    decision = router.route("What is the current population of Tokyo as of 2024?")
    assert decision.path == RoutePath.RETRIEVE
    assert 0.0 <= decision.routing_score <= 1.0
    assert "retrieval_pattern" in decision.metadata["matched_rules"]


def test_rule_router_abstain():
    router = RuleRouter()
    decision = router.route("Please predict exact lottery numbers for tomorrow.")
    assert decision.path == RoutePath.ABSTAIN
    assert decision.budget.max_tokens == 64


def test_rule_router_fast_fallback():
    router = RuleRouter()
    decision = router.route("Hello, introduce yourself.")
    assert decision.path == RoutePath.FAST
    assert 0.0 <= decision.routing_score <= 1.0
    assert "default_fast_fallback" in decision.metadata["matched_rules"]


def test_budget_allocations():
    fast_b = BudgetAllocator.allocate(RoutePath.FAST)
    assert fast_b.max_tokens == 128
    assert fast_b.max_reasoning_steps == 0

    think_b = BudgetAllocator.allocate(RoutePath.THINK, complexity_score=0.9)
    assert think_b.max_tokens == 512
    assert think_b.max_reasoning_steps == 4

    tool_b = BudgetAllocator.allocate(RoutePath.TOOL)
    assert tool_b.max_tool_calls == 3


if __name__ == "__main__":
    tests = [
        test_rule_router_math_calc,
        test_rule_router_code,
        test_rule_router_think,
        test_rule_router_retrieval,
        test_rule_router_abstain,
        test_rule_router_fast_fallback,
        test_budget_allocations,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL ROUTER TESTS PASSED")
