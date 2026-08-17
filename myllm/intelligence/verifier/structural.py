"""
myllm.intelligence.verifier.structural — Structural and repetition verifier.
"""

from typing import Optional, List, Dict, Any
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.schemas import VerificationStatus


def analyze_repetition(text: str) -> float:
    """
    Calculate a repetition score (0.0 to 1.0) based on repeating 3-grams.
    Returns 0.0 for diverse non-repetitive text, higher for degenerate loops.
    """
    words = text.split()
    if len(words) < 3:
        return 0.0

    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0

    unique_trigrams = set(trigrams)
    repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
    return repetition_ratio


class StructuralVerifier(BaseVerifier):
    """
    Verifies that the generated response is non-empty, within reasonable length bounds,
    and free from degenerative n-gram repetition loops.
    """

    def __init__(self, max_repetition_threshold: float = 0.25, min_words: int = 1):
        self.max_repetition_threshold = max_repetition_threshold
        self.min_words = min_words

    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        text = candidate_answer.strip()

        if not text:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=0.0,
                verifier_name="StructuralVerifier",
                critique="Candidate answer is empty or whitespace only.",
            )

        words = text.split()
        if len(words) < self.min_words:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=0.2,
                verifier_name="StructuralVerifier",
                critique=f"Candidate answer too short ({len(words)} words < {self.min_words}).",
            )

        rep_score = analyze_repetition(text)
        if rep_score > self.max_repetition_threshold:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=max(0.0, 1.0 - rep_score),
                verifier_name="StructuralVerifier",
                critique=f"Excessive repetitive loops detected (repetition score {rep_score:.2f} > {self.max_repetition_threshold:.2f}).",
                details={"repetition_score": rep_score},
            )

        return VerificationResult(
            status=VerificationStatus.PASS,
            score=0.95,
            verifier_name="StructuralVerifier",
            details={"repetition_score": rep_score, "word_count": len(words)},
        )
