"""
myllm.intelligence.telemetry — Structured telemetry, metrics, and trace collector.
Enables full observability of routing decisions, budgets, latencies, and verifier outcomes.
"""

from typing import Any, Dict, List, Optional
import time
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from myllm.intelligence.schemas import RouteDecision, ExecutionState, VerificationStatus, RoutePath


@dataclass
class TelemetryEvent:
    event_type: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """
    Collects execution trace events during query processing and generates structured logs.
    """
    def __init__(self, query_id: Optional[str] = None):
        self.query_id = query_id or str(uuid.uuid4())
        self.events: List[TelemetryEvent] = []

    def record_routing(self, decision: RouteDecision, query: str) -> None:
        self.events.append(
            TelemetryEvent(
                event_type="routing_decision",
                data={
                    "query_id": self.query_id,
                    "query_length": len(query),
                    "selected_path": decision.path.value,
                    "routing_score": decision.routing_score,
                    "reason": decision.reason,
                    "budget": decision.budget.to_dict(),
                    "metadata": decision.metadata,
                },
            )
        )

    def record_execution_step(self, step_name: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TelemetryEvent(
                event_type="execution_step",
                data={
                    "query_id": self.query_id,
                    "step": step_name,
                    "duration_ms": duration_ms,
                    "metadata": metadata or {},
                },
            )
        )

    def record_verification(self, status: VerificationStatus, score: float, critique: Optional[str] = None) -> None:
        self.events.append(
            TelemetryEvent(
                event_type="verification_outcome",
                data={
                    "query_id": self.query_id,
                    "status": status.value,
                    "score": score,
                    "critique": critique,
                },
            )
        )

    def build_summary(
        self,
        state: ExecutionState,
        final_status: VerificationStatus,
        tools_used: Optional[List[str]] = None,
        retrieval_used: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds the canonical telemetry payload required for training future learned routers.
        """
        # Success is defined as reaching PASS or acceptable UNVERIFIED without abort/crash
        is_success = (final_status in (VerificationStatus.PASS, VerificationStatus.UNVERIFIED)) and not state.is_aborted

        tools_list = tools_used or [res.tool_name for res in state.tool_history]

        return {
            "query_id": self.query_id,
            "route": state.route_decision.path.value,
            "routing_score": state.route_decision.routing_score,
            "latency_ms": round(state.elapsed_ms(), 2),
            "verification_status": final_status.value,
            "tools_used": tools_list,
            "retrieval_used": retrieval_used or (state.route_decision.path == RoutePath.RETRIEVE),
            "success": is_success,
            "tokens_consumed": state.tokens_consumed,
            "tool_calls_made": state.tool_calls_made,
            "retries_attempted": state.retries_attempted,
            "event_count": len(self.events),
            "events": [
                {
                    "type": e.event_type,
                    "timestamp": e.timestamp,
                    "data": e.data,
                }
                for e in self.events
            ],
        }

    def append_to_jsonl(self, summary: Dict[str, Any], filepath: str) -> None:
        """Appends structured telemetry summary to a persistent JSONL log file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
