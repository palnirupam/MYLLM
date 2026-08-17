"""
myllm.intelligence.schemas — Data models and schemas for the Dhruva Intelligence System.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


class RoutePath(str, Enum):
    FAST = "FAST"
    THINK = "THINK"
    TOOL = "TOOL"
    RETRIEVE = "RETRIEVE"
    ABSTAIN = "ABSTAIN"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    UNVERIFIED = "UNVERIFIED"
    REVISE = "REVISE"
    ABSTAIN = "ABSTAIN"


@dataclass
class ComputeBudget:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_reasoning_steps: int = 0
    max_tool_calls: int = 0
    max_retries: int = 1
    enable_verification: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_reasoning_steps": self.max_reasoning_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
            "enable_verification": self.enable_verification,
        }


@dataclass
class RouteDecision:
    path: RoutePath
    routing_score: float  # Heuristic score in [0.0, 1.0], not calibrated confidence
    reason: str
    budget: ComputeBudget
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.value,
            "routing_score": self.routing_score,
            "reason": self.reason,
            "budget": self.budget.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class Document:
    doc_id: str
    content: str
    score: float = 0.0
    source: str = "unknown"
    provenance_uri: Optional[str] = None
    language: str = "en"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "provenance_uri": self.provenance_uri,
            "language": self.language,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ToolError(Exception):
    """Raised when tool execution encounters an unrecoverable failure."""
    def __init__(self, tool_name: str, message: str, is_recoverable: bool = True):
        super().__init__(f"Tool [{tool_name}] error: {message}")
        self.tool_name = tool_name
        self.message = message
        self.is_recoverable = is_recoverable


@dataclass
class ExecutionState:
    """
    Tracks state throughout the lifecycle of a query across paths, retries, and verifiers.
    """
    query: str
    route_decision: RouteDecision
    budget: ComputeBudget
    tokens_consumed: int = 0
    reasoning_steps_taken: int = 0
    tool_calls_made: int = 0
    retries_attempted: int = 0
    escalation_path: Optional[RoutePath] = None
    tool_history: List[ToolResult] = field(default_factory=list)
    verification_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    is_aborted: bool = False
    abort_reason: Optional[str] = None

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000.0


@dataclass
class AnswerResult:
    """
    Public structured output returned to the user or caller.
    Does not leak raw internal scratchpads or chain-of-thought traces.
    """
    answer: str
    confidence: float
    route_taken: RoutePath
    verification_status: VerificationStatus
    evidence_citations: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    uncertainty_reason: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "route_taken": self.route_taken.value,
            "verification_status": self.verification_status.value,
            "evidence_citations": self.evidence_citations,
            "tools_used": self.tools_used,
            "uncertainty_reason": self.uncertainty_reason,
            "telemetry": self.telemetry,
        }
