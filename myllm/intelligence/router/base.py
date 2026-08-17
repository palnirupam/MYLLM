"""
myllm.intelligence.router.base — Abstract Base Router definition.
"""

from abc import ABC, abstractmethod
from typing import Optional
from myllm.intelligence.schemas import RouteDecision


class BaseRouter(ABC):
    """
    Abstract interface for task routing.
    Estimates complexity, evidence requirement, tool requirement, and assigns compute budgets.
    """

    @abstractmethod
    def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        """
        Classifies incoming query and determines the execution path and budget.

        Args:
            query: The user request or prompt.
            context: Optional prior conversation or grounding context.

        Returns:
            RouteDecision containing the selected path, routing score (0.0 to 1.0),
            reason, and ComputeBudget.
        """
        pass
