"""
tests/unit/test_schemas_policy.py
Verifies schemas, enums, and StateManager policy rules for Dhruva Intelligence System.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.schemas import (
    RoutePath,
    VerificationStatus,
    ComputeBudget,
    RouteDecision,
    ExecutionState,
    AnswerResult,
    ToolCall,
    ToolResult,
)
from myllm.intelligence.policy import StateManager


def test_schemas_enums_and_structures():
    # Verify all enum values exist
    assert RoutePath.FAST == "FAST"
    assert RoutePath.THINK == "THINK"
    assert RoutePath.TOOL == "TOOL"
    assert RoutePath.RETRIEVE == "RETRIEVE"
    assert RoutePath.ABSTAIN == "ABSTAIN"

    assert VerificationStatus.PASS == "PASS"
    assert VerificationStatus.UNVERIFIED == "UNVERIFIED"
    assert VerificationStatus.REVISE == "REVISE"
    assert VerificationStatus.ABSTAIN == "ABSTAIN"

    budget = ComputeBudget(max_tokens=256, max_tool_calls=2)
    assert budget.max_tokens == 256
    assert budget.max_tool_calls == 2
    assert isinstance(budget.to_dict(), dict)

    decision = RouteDecision(
        path=RoutePath.FAST,
        routing_score=0.85,
        reason="Test greeting",
        budget=budget,
    )
    assert decision.path == RoutePath.FAST
    assert 0.0 <= decision.routing_score <= 1.0


def test_state_manager_tool_limits():
    manager = StateManager()
    budget = ComputeBudget(max_tool_calls=2)
    decision = RouteDecision(path=RoutePath.TOOL, routing_score=0.9, reason="Test", budget=budget)
    state = manager.create_initial_state("Compute 2+2", decision)

    # 1st call
    can_call, err = manager.can_invoke_tool(state)
    assert can_call is True
    assert err is None
    manager.record_tool_invocation(state)

    # 2nd call
    can_call, err = manager.can_invoke_tool(state)
    assert can_call is True
    manager.record_tool_invocation(state)

    # 3rd call — should be blocked
    can_call, err = manager.can_invoke_tool(state)
    assert can_call is False
    assert "Exceeded max tool calls" in err


def test_state_manager_retry_limits():
    manager = StateManager()
    budget = ComputeBudget(max_retries=1)
    decision = RouteDecision(path=RoutePath.THINK, routing_score=0.8, reason="Test", budget=budget)
    state = manager.create_initial_state("Explain relativity", decision)

    can_retry, err = manager.can_retry(state)
    assert can_retry is True
    manager.record_retry(state)

    # Second retry should be blocked
    can_retry, err = manager.can_retry(state)
    assert can_retry is False
    assert "Exceeded max retries" in err


def test_state_manager_escalation_rules():
    manager = StateManager()
    budget = ComputeBudget(max_retries=1)
    decision = RouteDecision(path=RoutePath.FAST, routing_score=0.7, reason="Fast", budget=budget)
    state = manager.create_initial_state("Who is current PM?", decision)

    # FastPath + UNVERIFIED -> escalate to RETRIEVE
    should_esc, target_path, reason = manager.should_escalate(state, VerificationStatus.UNVERIFIED)
    assert should_esc is True
    assert target_path == RoutePath.RETRIEVE

    # FastPath + REVISE -> escalate to THINK
    should_esc, target_path, reason = manager.should_escalate(state, VerificationStatus.REVISE)
    assert should_esc is True
    assert target_path == RoutePath.THINK

    # If retries exceeded, force abstention
    manager.record_retry(state)
    should_esc, target_path, reason = manager.should_escalate(state, VerificationStatus.REVISE)
    assert should_esc is False
    assert target_path == RoutePath.ABSTAIN


def test_state_manager_token_ceiling_abort():
    manager = StateManager()
    budget = ComputeBudget(max_tokens=100)
    decision = RouteDecision(path=RoutePath.FAST, routing_score=0.7, reason="Fast", budget=budget)
    state = manager.create_initial_state("Test prompt", decision)

    manager.record_tokens(state, 50)
    assert state.is_aborted is False

    # Exceeding 2x budget aborts state
    manager.record_tokens(state, 200)
    assert state.is_aborted is True
    assert "Token budget exceeded" in state.abort_reason


if __name__ == "__main__":
    tests = [
        test_schemas_enums_and_structures,
        test_state_manager_tool_limits,
        test_state_manager_retry_limits,
        test_state_manager_escalation_rules,
        test_state_manager_token_ceiling_abort,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL SCHEMAS & POLICY TESTS PASSED")
