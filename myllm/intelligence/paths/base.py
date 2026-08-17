"""
myllm.intelligence.paths.base — Abstract Base Path and PathOutput definition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from myllm.intelligence.schemas import ExecutionState
from myllm.runtime.interfaces.base import InferenceRuntime


@dataclass
class PathOutput:
    """
    Standardized internal output returned by an execution path before verification.
    """
    text: str
    tokens_generated: int
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    confidence_estimate: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePath(ABC):
    """
    Abstract interface for all query execution paths.
    """

    @abstractmethod
    def execute(
        self,
        query: str,
        state: ExecutionState,
        runtime: InferenceRuntime,
    ) -> PathOutput:
        """
        Executes the query under the allocated state budget.
        """
        pass
