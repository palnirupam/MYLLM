"""
myllm.intelligence.verifier.factual — Evidence-grounded factual consistency verifier with entity attribution.
"""

import re
from typing import Optional, List, Dict, Any
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.schemas import VerificationStatus

_QUESTION_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "cite", "sources", "source", "according", "tell", "about", "please",
    "the", "is", "are", "was", "were", "been", "have", "has", "had", "for", "with"
}


class EvidenceGroundedVerifier(BaseVerifier):
    """
    Validates candidate factual assertions against retrieved grounding evidence.
    Tracks supported_claims, unsupported_claims, evidence_ids, verification_score,
    and performs exact entity-attribution checks.
    """

    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        if not evidence or len(evidence) == 0:
            # Under Dhruva truthfulness policy, zero evidence means factual claims cannot be certified
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                score=0.5,
                verifier_name="EvidenceGroundedVerifier",
                critique="No external grounded evidence provided; factual claim remains unverified.",
                details={
                    "evidence_count": 0,
                    "supported_claims": [],
                    "unsupported_claims": [candidate_answer.strip()],
                    "evidence_ids": [],
                    "entity_attribution_passed": False,
                    "verification_score": 0.5,
                },
            )

        combined_evidence = " ".join(evidence).lower()

        # 1. Entity Attribution Check (Requirement 8):
        # Extract queried entities: prioritize capitalized proper nouns, fallback to non-stopwords
        raw_words = [w.strip(".,;:?!\"'()") for w in query.split() if len(w.strip(".,;:?!\"'()")) > 1]
        capitalized_entities = [
            w.lower() for idx, w in enumerate(raw_words)
            if w and w[0].isupper() and (idx > 0 or w.lower() not in _QUESTION_STOPWORDS) and w.lower() not in _QUESTION_STOPWORDS
        ]

        target_entities = capitalized_entities if capitalized_entities else [
            w.lower() for w in raw_words if w.lower() not in _QUESTION_STOPWORDS and len(w) > 3
        ]

        entity_attribution_passed = True
        if target_entities:
            entity_matches = sum(1 for e in target_entities if e in combined_evidence)
            if entity_matches == 0:
                entity_attribution_passed = False

        # Extract evidence IDs if present in evidence envelopes
        doc_id_matches = re.findall(r"doc_id=[\"']([^\"']+)[\"']|\[(?:Doc|Source):\s*([^\]\|]+)", " ".join(evidence))
        evidence_ids = [m[0] or m[1] for m in doc_id_matches if (m[0] or m[1])]

        # 2. Claim-level Overlap Check:
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", candidate_answer) if len(s.strip()) > 5]
        if not sentences:
            sentences = [candidate_answer.strip()]

        supported_claims = []
        unsupported_claims = []

        for sent in sentences:
            sent_words = [w.strip(".,;:?!\"'()").lower() for w in sent.split() if len(w) > 3]
            if not sent_words:
                continue

            matches = sum(1 for w in sent_words if w in combined_evidence)
            match_ratio = matches / len(sent_words)

            if match_ratio >= 0.35:
                supported_claims.append(sent)
            else:
                unsupported_claims.append(sent)

        total_claims = len(supported_claims) + len(unsupported_claims)
        support_rate = len(supported_claims) / max(1, total_claims)

        details = {
            "evidence_count": len(evidence),
            "evidence_ids": evidence_ids,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "entity_attribution_passed": entity_attribution_passed,
            "verification_score": round(support_rate, 3),
        }

        # If entity attribution failed, evidence is only topically related -> REVISE
        if not entity_attribution_passed:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=0.2,
                verifier_name="EvidenceGroundedVerifier",
                critique="Entity attribution failure: evidence does not mention the queried subject.",
                details=details,
            )

        if support_rate >= 0.50:
            return VerificationResult(
                status=VerificationStatus.PASS,
                score=min(1.0, 0.6 + 0.4 * support_rate),
                verifier_name="EvidenceGroundedVerifier",
                details=details,
            )
        else:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=support_rate,
                verifier_name="EvidenceGroundedVerifier",
                critique=f"Insufficient grounding ({len(unsupported_claims)} of {total_claims} claims unsupported by evidence).",
                details=details,
            )
