"""
myllm.intelligence.verifier.base — Base Verifier contract and result dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from myllm.intelligence.schemas import VerificationStatus


@dataclass
class VerificationResult:
    status: VerificationStatus
    score: float  # Score in [0.0, 1.0]
    verifier_name: str
    critique: Optional[str] = None
    suggested_revision: Optional[str] = None
    details: Dict[str, Any] = None


class BaseVerifier(ABC):
    """
    Abstract verifier interface for validating candidate outputs.
    """

    @abstractmethod
    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        """
        Evaluates candidate answer against specific quality, correctness, or grounding rules.
        """
        pass
