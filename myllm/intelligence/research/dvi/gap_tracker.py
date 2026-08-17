"""
myllm.intelligence.research.dvi.gap_tracker — Tracks domain capability gaps from verification outcomes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class DomainFailureRecord:
    query: str
    domain: str
    failure_type: str
    critique: str
    timestamp: float = field(default_factory=time.time)


class CapabilityGapTracker:
    """
    Records and aggregates capability failures and verifier rejections to identify
    underperforming knowledge or reasoning domains.
    """

    def __init__(self):
        self.failures: List[DomainFailureRecord] = []
        self.domain_counts: Dict[str, int] = {}

    def record_failure(self, query: str, domain: str, failure_type: str, critique: str) -> None:
        rec = DomainFailureRecord(query=query, domain=domain, failure_type=failure_type, critique=critique)
        self.failures.append(rec)
        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1

    def get_gap_summary(self) -> Dict[str, Any]:
        return {
            "total_recorded_failures": len(self.failures),
            "domain_failure_breakdown": self.domain_counts,
        }
