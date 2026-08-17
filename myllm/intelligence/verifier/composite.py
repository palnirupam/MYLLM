"""
myllm.intelligence.verifier.composite — Composite verifier orchestrating specialized checks.
"""

from typing import Optional, List, Dict, Any
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.verifier.structural import StructuralVerifier
from myllm.intelligence.verifier.mathematical import MathematicalVerifier
from myllm.intelligence.verifier.code import CodeExecutionVerifier
from myllm.intelligence.verifier.factual import EvidenceGroundedVerifier
from myllm.intelligence.schemas import VerificationStatus


class CompositeVerifier(BaseVerifier):
    """
    Sequentially executes structural, mathematical, code execution, and evidence-grounded verifiers.
    """

    def __init__(
        self,
        structural_verifier: Optional[StructuralVerifier] = None,
        mathematical_verifier: Optional[MathematicalVerifier] = None,
        code_verifier: Optional[CodeExecutionVerifier] = None,
        factual_verifier: Optional[EvidenceGroundedVerifier] = None,
    ):
        self.structural_verifier = structural_verifier or StructuralVerifier()
        self.mathematical_verifier = mathematical_verifier or MathematicalVerifier()
        self.code_verifier = code_verifier or CodeExecutionVerifier()
        self.factual_verifier = factual_verifier or EvidenceGroundedVerifier()

    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        # 1. Structural check (must not be empty or looping)
        res = self.structural_verifier.verify(query, candidate_answer, evidence, tool_results)
        if res.status != VerificationStatus.PASS:
            return res

        # 2. Math check (if arithmetic expressions exist)
        math_res = self.mathematical_verifier.verify(query, candidate_answer, evidence, tool_results)
        if math_res.status == VerificationStatus.REVISE:
            return math_res

        # 3. Code check (if code blocks exist)
        code_res = self.code_verifier.verify(query, candidate_answer, evidence, tool_results)
        if code_res.status == VerificationStatus.REVISE:
            return code_res

        # 4. Factual check
        FACTUAL_QUERY_PATTERN = [
            "who ", "who was", "who is", "when was", "where is", "where was",
            "invented", "discovered", "capital of", "born", "died", "founded",
            "population", "author of", "president", "prime minister", "history of"
        ]
        is_factual_query = any(pattern in query.lower() for pattern in FACTUAL_QUERY_PATTERN)

        if evidence is not None and len(evidence) > 0:
            fact_res = self.factual_verifier.verify(query, candidate_answer, evidence, tool_results)
            if fact_res.status != VerificationStatus.PASS:
                return fact_res
        elif is_factual_query:
            # Factual query with zero evidence must be UNVERIFIED under truthfulness contract
            return self.factual_verifier.verify(query, candidate_answer, evidence=None)

        # Return pass with details
        return VerificationResult(
            status=VerificationStatus.PASS,
            score=min(res.score, 1.0),
            verifier_name="CompositeVerifier",
            details={
                "structural": res.status.value,
                "mathematical": math_res.status.value,
                "code": code_res.status.value,
                "factual": "EVALUATED" if (evidence and len(evidence) > 0) else ("UNVERIFIED_NO_EVIDENCE" if is_factual_query else "SKIPPED_NON_FACTUAL"),
            },
        )
