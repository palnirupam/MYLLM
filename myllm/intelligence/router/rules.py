"""
myllm.intelligence.router.rules — Deterministic rule-based baseline router.
Categorizes queries using pattern matching and lexical heuristics.
Emits routing_score (heuristic in [0.0, 1.0]) and structured metadata.
"""

import re
from typing import Optional, Dict, Any
from myllm.intelligence.router.base import BaseRouter
from myllm.intelligence.router.budget import BudgetAllocator
from myllm.intelligence.schemas import RouteDecision, RoutePath


class RuleRouter(BaseRouter):
    """
    Deterministic rule-based router serving as the baseline for routing decisions.
    Emits heuristic routing_score and logs pattern triggers for future learned router training.
    """

    # Calculator / Arithmetic patterns
    MATH_CALC_REGEX = re.compile(
        r"(\d+\s*[\+\-\*\/\^\%]\s*\d+)|"
        r"(calculate|compute|solve)\s+(\d+|the\s+(sum|product|integral|derivative|value|equation|area))|"
        r"(what is\s+\d+)",
        re.IGNORECASE
    )

    # Code execution / debugging patterns
    CODE_REGEX = re.compile(
        r"(def\s+\w+\s*\(|class\s+\w+|import\s+\w+|```|write a\s+.*(function|script|program|code)|python|javascript|debug this|fix the bug)",
        re.IGNORECASE
    )

    # Deep reasoning / multi-step reasoning triggers
    THINK_REGEX = re.compile(
        r"(step[\s-]by[\s-]step|prove that|logic puzzle|riddle|compare and contrast in detail|why does|design a system|derive the formula)",
        re.IGNORECASE
    )

    # Evidence / Retrieval triggers (current events, specific entity lookups, citations, factual inquiries)
    RETRIEVAL_REGEX = re.compile(
        r"(latest news|current population|who is the current|as of \d{4}|cite sources|according to|search for|"
        r"who (developed|discovered|invented|founded|wrote|created)|"
        r"when was|where was|what year was|boiling point|melting point)",
        re.IGNORECASE
    )

    # Obvious unanswerable / nonsensical / adversarial triggers
    ABSTAIN_REGEX = re.compile(
        r"(predict exact lottery numbers|what am i thinking right now|tell me a secret confidential key)",
        re.IGNORECASE
    )

    def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        cleaned = query.strip()
        metadata: Dict[str, Any] = {
            "query_char_count": len(cleaned),
            "query_word_count": len(cleaned.split()),
            "matched_rules": [],
        }

        # 1. Check explicit abstention patterns
        if self.ABSTAIN_REGEX.search(cleaned):
            metadata["matched_rules"].append("abstain_pattern")
            return RouteDecision(
                path=RoutePath.ABSTAIN,
                routing_score=0.90,
                reason="Query matched explicit abstention pattern (unanswerable/speculative)",
                budget=BudgetAllocator.allocate(RoutePath.ABSTAIN),
                metadata=metadata,
            )

        # 2. Check Tool / Math Calculation patterns
        if self.MATH_CALC_REGEX.search(cleaned):
            metadata["matched_rules"].append("math_calc_pattern")
            return RouteDecision(
                path=RoutePath.TOOL,
                routing_score=0.85,
                reason="Query contains arithmetic calculation or explicit math problem",
                budget=BudgetAllocator.allocate(RoutePath.TOOL, complexity_score=0.6),
                metadata=metadata,
            )

        # 3. Check Code patterns
        if self.CODE_REGEX.search(cleaned):
            metadata["matched_rules"].append("code_execution_pattern")
            return RouteDecision(
                path=RoutePath.TOOL,
                routing_score=0.80,
                reason="Query involves code generation, debugging, or execution",
                budget=BudgetAllocator.allocate(RoutePath.TOOL, complexity_score=0.8),
                metadata=metadata,
            )

        # 4. Check Retrieval patterns
        if self.RETRIEVAL_REGEX.search(cleaned):
            metadata["matched_rules"].append("retrieval_pattern")
            return RouteDecision(
                path=RoutePath.RETRIEVE,
                routing_score=0.75,
                reason="Query requires grounded external facts or current information",
                budget=BudgetAllocator.allocate(RoutePath.RETRIEVE, complexity_score=0.7),
                metadata=metadata,
            )

        # 5. Check Deep Reasoning patterns
        if self.THINK_REGEX.search(cleaned):
            metadata["matched_rules"].append("deep_think_pattern")
            return RouteDecision(
                path=RoutePath.THINK,
                routing_score=0.70,
                reason="Query requires step-by-step reasoning or mathematical derivation",
                budget=BudgetAllocator.allocate(RoutePath.THINK, complexity_score=0.75),
                metadata=metadata,
            )

        # 6. Default to FAST path (single forward pass) for general queries
        metadata["matched_rules"].append("default_fast_fallback")
        return RouteDecision(
            path=RoutePath.FAST,
            routing_score=0.65,
            reason="Standard conversational or concise query routed to FastPath",
            budget=BudgetAllocator.allocate(RoutePath.FAST, complexity_score=0.3),
            metadata=metadata,
        )
