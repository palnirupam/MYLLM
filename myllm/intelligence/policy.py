"""
myllm.intelligence.policy — State and Policy Manager for Dhruva.
Controls execution limits, budget consumption, escalation rules, and abstention triggers.
"""

from typing import Optional, Tuple
from myllm.intelligence.schemas import (
    ExecutionState,
    RoutePath,
    VerificationStatus,
    ComputeBudget,
    RouteDecision,
)


class StateManager:
    """
    State and Policy Manager governing transitions between paths, tool execution,
    retries, and verification.
    """

    def __init__(self, default_budget: Optional[ComputeBudget] = None):
        self.default_budget = default_budget or ComputeBudget()

    def create_initial_state(self, query: str, decision: RouteDecision) -> ExecutionState:
        """Initializes a new execution state tracking budget and progress."""
        return ExecutionState(
            query=query,
            route_decision=decision,
            budget=decision.budget,
        )

    def check_execution_time(self, state: ExecutionState, max_time_ms: float = 10000.0) -> Tuple[bool, Optional[str]]:
        """Checks if total execution time has exceeded the safety timeout."""
        if state.elapsed_ms() > max_time_ms:
            state.is_aborted = True
            state.abort_reason = f"Execution timeout exceeded ({state.elapsed_ms():.1f}ms > {max_time_ms}ms)"
            return False, state.abort_reason
        return True, None

    def can_invoke_tool(self, state: ExecutionState, max_time_ms: float = 10000.0) -> Tuple[bool, Optional[str]]:
        """Validates if another tool invocation is allowed under current policy."""
        time_ok, time_err = self.check_execution_time(state, max_time_ms=max_time_ms)
        if not time_ok:
            return False, time_err

        if state.is_aborted:
            return False, f"Execution aborted: {state.abort_reason}"

        if state.tool_calls_made >= state.budget.max_tool_calls:
            return False, f"Exceeded max tool calls limit ({state.budget.max_tool_calls})"

        return True, None

    def record_tool_invocation(self, state: ExecutionState) -> None:
        """Increments tool invocation counter."""
        state.tool_calls_made += 1

    def can_retry(self, state: ExecutionState, max_time_ms: float = 10000.0) -> Tuple[bool, Optional[str]]:
        """Validates if a revision/retry pass is permitted."""
        time_ok, time_err = self.check_execution_time(state, max_time_ms=max_time_ms)
        if not time_ok:
            return False, time_err

        if state.is_aborted:
            return False, f"Execution aborted: {state.abort_reason}"

        if state.retries_attempted >= state.budget.max_retries:
            return False, f"Exceeded max retries limit ({state.budget.max_retries})"

        return True, None

    def record_retry(self, state: ExecutionState) -> None:
        """Increments retry counter."""
        state.retries_attempted += 1

    def record_tokens(self, state: ExecutionState, tokens_generated: int) -> None:
        """Records token generation expenditure."""
        state.tokens_consumed += tokens_generated
        if state.tokens_consumed > (state.budget.max_tokens * 2):
            # Hard safety ceiling to prevent runaway context
            state.is_aborted = True
            state.abort_reason = f"Token budget exceeded ({state.tokens_consumed} > {state.budget.max_tokens * 2})"

    def should_escalate(
        self,
        state: ExecutionState,
        verification_status: VerificationStatus,
        verification_critique: Optional[str] = None,
    ) -> Tuple[bool, Optional[RoutePath], str]:
        """
        Determines whether to escalate the query to a higher-compute or grounded path.
        For example:
        - FAST path returns UNVERIFIED on factual query -> escalate to RETRIEVE
        - FAST path returns REVISE -> escalate to THINK
        - THINK or RETRIEVE exceeded retries -> ABSTAIN
        """
        if state.is_aborted:
            return False, RoutePath.ABSTAIN, state.abort_reason or "State aborted"

        # If already at max retries, do not escalate further; force abstention or output
        if state.retries_attempted >= state.budget.max_retries:
            if verification_status in (VerificationStatus.REVISE, VerificationStatus.ABSTAIN):
                return False, RoutePath.ABSTAIN, "Max retries reached without passing verification."

        if verification_status == VerificationStatus.UNVERIFIED:
            # If query required evidence but FastPath was taken, escalate to RETRIEVE
            if state.route_decision.path == RoutePath.FAST:
                return True, RoutePath.RETRIEVE, "Fast path unverified on factual claim; escalating to retrieval."

        if verification_status == VerificationStatus.REVISE:
            if state.route_decision.path == RoutePath.FAST:
                return True, RoutePath.THINK, "Fast path failed verification critique; escalating to think path."

        return False, None, "No escalation required."

    def force_abstention(self, state: ExecutionState, reason: str) -> None:
        """Marks execution as aborted with explicit abstention reason."""
        state.is_aborted = True
        state.abort_reason = reason
        state.escalation_path = RoutePath.ABSTAIN
