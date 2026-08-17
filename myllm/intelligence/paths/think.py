"""
myllm.intelligence.paths.think — Bounded Adaptive Reasoning Path.
Performs bounded internal candidate generation, verifier critique checks,
and at most one revision pass without exposing hidden reasoning traces.
"""

from typing import Optional, List, Dict, Any
import time
from myllm.intelligence.paths.base import BasePath, PathOutput
from myllm.intelligence.schemas import ExecutionState, VerificationStatus
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.verifier.composite import CompositeVerifier
from myllm.runtime.interfaces.base import InferenceRuntime


class ThinkPath(BasePath):
    """
    Bounded adaptive reasoning path.
    Coordinates candidate generation, critique-driven internal revision, and verification.
    """

    def __init__(self, verifier: Optional[BaseVerifier] = None):
        self.verifier = verifier or CompositeVerifier()

    def execute(
        self,
        query: str,
        state: ExecutionState,
        runtime: InferenceRuntime,
    ) -> PathOutput:
        budget = state.budget
        max_steps = min(3, budget.max_reasoning_steps or 2)
        start_time = time.time()
        max_time_s = 5.0

        # Step 1: Initial candidate generation (Internal deliberation pass 1)
        state.reasoning_steps_taken += 1
        prompt = f"Question: {query}\nProvide a clear, rigorous, step-by-step verified response.\nAnswer:"

        candidate_raw = runtime.generate(
            prompt=prompt,
            max_new_tokens=min(budget.max_tokens, 256),
            temperature=budget.temperature,
            top_k=budget.top_k,
            top_p=budget.top_p,
        )

        if candidate_raw.startswith(prompt):
            candidate_text = candidate_raw[len(prompt):].strip()
        else:
            candidate_text = candidate_raw.strip()

        est_tokens = max(1, len(candidate_text.split()))
        state.tokens_consumed += est_tokens

        # Step 2: Internal Verification Check
        verdict: VerificationResult = self.verifier.verify(query=query, candidate_answer=candidate_text)

        # Step 3: At most ONE bounded revision attempt if verifier requests REVISE
        if verdict.status == VerificationStatus.REVISE and state.reasoning_steps_taken < max_steps:
            if (time.time() - start_time) < max_time_s:
                state.reasoning_steps_taken += 1
                state.retries_attempted += 1

                critique_msg = verdict.critique or "Please ensure logical consistency and eliminate repetition."
                revision_prompt = (
                    f"Question: {query}\n"
                    f"Draft Answer: {candidate_text}\n"
                    f"Correction Note: {critique_msg}\n"
                    f"Revised Final Answer:"
                )

                revised_raw = runtime.generate(
                    prompt=revision_prompt,
                    max_new_tokens=min(budget.max_tokens, 256),
                    temperature=max(0.1, budget.temperature - 0.2),
                    top_k=budget.top_k,
                    top_p=budget.top_p,
                )

                if revised_raw.startswith(revision_prompt):
                    candidate_text = revised_raw[len(revision_prompt):].strip()
                else:
                    candidate_text = revised_raw.strip()

                revised_tokens = max(1, len(candidate_text.split()))
                state.tokens_consumed += revised_tokens

                # Re-verify revised candidate
                verdict = self.verifier.verify(query=query, candidate_answer=candidate_text)

        # Strip any accidental private thinking tags if present
        clean_final_text = candidate_text.replace("<scratchpad>", "").replace("</scratchpad>", "").strip()

        return PathOutput(
            text=clean_final_text,
            tokens_generated=state.tokens_consumed,
            confidence_estimate=verdict.score,
            metadata={
                "path_type": "THINK",
                "reasoning_steps_taken": state.reasoning_steps_taken,
                "retries_attempted": state.retries_attempted,
                "internal_verifier_status": verdict.status.value,
            },
        )
