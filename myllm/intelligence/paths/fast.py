"""
myllm.intelligence.paths.fast — Fast Path execution.
Provides lowest-latency single-pass generation for simple, formatting, and conversational queries.
"""

from myllm.intelligence.paths.base import BasePath, PathOutput
from myllm.intelligence.schemas import ExecutionState
from myllm.runtime.interfaces.base import InferenceRuntime


class FastPath(BasePath):
    """
    Executes a single forward generation pass with no tool invocation or recursive loops.
    """

    def execute(
        self,
        query: str,
        state: ExecutionState,
        runtime: InferenceRuntime,
    ) -> PathOutput:
        budget = state.budget

        # Generate text using runtime
        generated_text = runtime.generate(
            prompt=query,
            max_new_tokens=budget.max_tokens,
            temperature=budget.temperature,
            top_k=budget.top_k,
            top_p=budget.top_p,
        )

        # Strip prompt if returned with prompt prefix
        if generated_text.startswith(query):
            answer_text = generated_text[len(query):].strip()
        else:
            answer_text = generated_text.strip()

        # Estimate tokens generated (approx 1 token ~= 4 chars or split)
        # Runtime could report exact, but length fallback is safe
        est_tokens = max(1, len(answer_text.split()))

        state.tokens_consumed += est_tokens

        return PathOutput(
            text=answer_text,
            tokens_generated=est_tokens,
            confidence_estimate=0.85,
            metadata={"path_type": "FAST", "single_pass": True},
        )
