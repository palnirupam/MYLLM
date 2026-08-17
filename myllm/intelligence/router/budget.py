"""
myllm.intelligence.router.budget — ComputeBudget allocator.
"""

from myllm.intelligence.schemas import ComputeBudget, RoutePath


class BudgetAllocator:
    """
    Allocates tailored inference compute budgets based on the chosen execution path
    and estimated complexity.
    """

    @staticmethod
    def allocate(path: RoutePath, complexity_score: float = 0.5) -> ComputeBudget:
        if path == RoutePath.FAST:
            return ComputeBudget(
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_reasoning_steps=0,
                max_tool_calls=0,
                max_retries=1,
                enable_verification=True,
            )

        elif path == RoutePath.THINK:
            # Scale tokens and reasoning steps based on complexity
            max_tokens = 256 if complexity_score < 0.7 else 512
            reasoning_steps = 2 if complexity_score < 0.7 else 4
            return ComputeBudget(
                max_tokens=max_tokens,
                temperature=0.4,  # Lower temperature for deterministic reasoning
                top_p=0.9,
                top_k=40,
                max_reasoning_steps=reasoning_steps,
                max_tool_calls=0,
                max_retries=2,
                enable_verification=True,
            )

        elif path == RoutePath.TOOL:
            return ComputeBudget(
                max_tokens=256,
                temperature=0.2,  # Low temperature for syntax/parameters
                top_p=0.9,
                top_k=30,
                max_reasoning_steps=2,
                max_tool_calls=3,
                max_retries=2,
                enable_verification=True,
            )

        elif path == RoutePath.RETRIEVE:
            return ComputeBudget(
                max_tokens=256,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                max_reasoning_steps=1,
                max_tool_calls=1,  # Retrieval call count
                max_retries=1,
                enable_verification=True,
            )

        else:  # ABSTAIN or default
            return ComputeBudget(
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_reasoning_steps=0,
                max_tool_calls=0,
                max_retries=0,
                enable_verification=False,
            )
