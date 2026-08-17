"""
myllm.intelligence.orchestrator — Central Orchestrator for the Dhruva Compound AI System.
Coordinates Router -> StateManager -> Execution Path (Fast/Tool/Retrieve/Think) -> Verifiers -> Telemetry -> AnswerResult.
"""

from typing import Optional, Dict, Any, List
from myllm.intelligence.schemas import (
    AnswerResult,
    RoutePath,
    VerificationStatus,
    RouteDecision,
    ExecutionState,
)
from myllm.intelligence.policy import StateManager
from myllm.intelligence.router.base import BaseRouter
from myllm.intelligence.router.rules import RuleRouter
from myllm.intelligence.paths.fast import FastPath
from myllm.intelligence.paths.tool_path import ToolPath
from myllm.intelligence.paths.retrieve_path import RetrievePath
from myllm.intelligence.paths.think import ThinkPath
from myllm.intelligence.tools.base import ToolRegistry
from myllm.intelligence.tools.calculator import SafeCalculatorTool
from myllm.intelligence.tools.python_repl import SandboxedPythonTool
from myllm.intelligence.retrieval.base import BaseRetriever, BaseContextBuilder
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.intelligence.retrieval.context_builder import StructuredContextBuilder
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.verifier.composite import CompositeVerifier
from myllm.intelligence.telemetry import TelemetryCollector
from myllm.runtime.interfaces.base import InferenceRuntime


class DhruvaOrchestrator:
    """
    Central orchestration engine managing query dispatch, state policies,
    tool execution, evidence retrieval, bounded reasoning, verification pipelines,
    and structured answer generation.
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        router: Optional[BaseRouter] = None,
        state_manager: Optional[StateManager] = None,
        verifier: Optional[BaseVerifier] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retriever: Optional[BaseRetriever] = None,
        context_builder: Optional[BaseContextBuilder] = None,
        telemetry_log_file: Optional[str] = None,
    ):
        self.runtime = runtime
        self.router = router or RuleRouter()
        self.state_manager = state_manager or StateManager()
        self.verifier = verifier or CompositeVerifier()
        self.telemetry_log_file = telemetry_log_file

        # Initialize tools
        self.tool_registry = tool_registry or ToolRegistry()
        if not self.tool_registry.list_tools():
            self.tool_registry.register(SafeCalculatorTool())
            self.tool_registry.register(SandboxedPythonTool())

        # Initialize retrieval
        self.retriever = retriever or InMemoryBM25Retriever()
        self.context_builder = context_builder or StructuredContextBuilder()

        # Initialize execution paths
        self.fast_path = FastPath()
        self.tool_path = ToolPath(self.tool_registry)
        self.retrieve_path = RetrievePath(self.retriever, self.context_builder)
        self.think_path = ThinkPath(self.verifier)

    def answer(
        self,
        query: str,
        context: Optional[str] = None,
        evidence: Optional[List[str]] = None,
    ) -> AnswerResult:
        telemetry = TelemetryCollector()

        # 1. Routing step
        decision: RouteDecision = self.router.route(query, context=context)
        telemetry.record_routing(decision, query)

        # 2. State initialization
        state: ExecutionState = self.state_manager.create_initial_state(query, decision)

        # 3. Handle explicit Abstention
        if decision.path == RoutePath.ABSTAIN:
            telemetry.record_verification(VerificationStatus.ABSTAIN, score=0.0, critique=decision.reason)
            summary = telemetry.build_summary(state, VerificationStatus.ABSTAIN, tools_used=[], retrieval_used=False)
            if self.telemetry_log_file:
                telemetry.append_to_jsonl(summary, self.telemetry_log_file)

            return AnswerResult(
                answer="I do not have sufficient verifiable information to answer this question reliably.",
                confidence=0.0,
                route_taken=RoutePath.ABSTAIN,
                verification_status=VerificationStatus.ABSTAIN,
                uncertainty_reason=decision.reason,
                telemetry=summary,
            )

        # 4. Dispatch Path
        if decision.path == RoutePath.FAST:
            output = self.fast_path.execute(query, state, self.runtime)

            # Run Verifiers
            verdict: VerificationResult = self.verifier.verify(
                query=query,
                candidate_answer=output.text,
                evidence=evidence or output.evidence,
            )
            telemetry.record_verification(verdict.status, score=verdict.score, critique=verdict.critique)

            # Escalation check on failure: FAST -> THINK
            if verdict.status == VerificationStatus.REVISE:
                should_esc, target, rsn = self.state_manager.should_escalate(state, verdict.status, verdict.critique)
                if should_esc and target == RoutePath.THINK:
                    state.escalation_path = RoutePath.THINK
                    output = self.think_path.execute(query, state, self.runtime)
                    verdict = self.verifier.verify(query=query, candidate_answer=output.text, evidence=evidence)
                    telemetry.record_verification(verdict.status, score=verdict.score, critique=verdict.critique)

                    if verdict.status == VerificationStatus.REVISE:
                        summary = telemetry.build_summary(state, VerificationStatus.ABSTAIN)
                        if self.telemetry_log_file:
                            telemetry.append_to_jsonl(summary, self.telemetry_log_file)
                        return AnswerResult(
                            answer="I am uncertain about this answer as it could not be verified.",
                            confidence=0.0,
                            route_taken=RoutePath.THINK,
                            verification_status=VerificationStatus.ABSTAIN,
                            uncertainty_reason=verdict.critique or rsn,
                            telemetry=summary,
                        )

                elif target == RoutePath.ABSTAIN or verdict.status == VerificationStatus.REVISE:
                    summary = telemetry.build_summary(state, VerificationStatus.ABSTAIN)
                    if self.telemetry_log_file:
                        telemetry.append_to_jsonl(summary, self.telemetry_log_file)
                    return AnswerResult(
                        answer="I am uncertain about this answer as it could not be verified.",
                        confidence=0.0,
                        route_taken=decision.path,
                        verification_status=VerificationStatus.ABSTAIN,
                        uncertainty_reason=verdict.critique or rsn,
                        telemetry=summary,
                    )

            final_route = state.escalation_path or RoutePath.FAST
            summary = telemetry.build_summary(state, verdict.status, tools_used=[], retrieval_used=False)
            if self.telemetry_log_file:
                telemetry.append_to_jsonl(summary, self.telemetry_log_file)

            return AnswerResult(
                answer=output.text,
                confidence=verdict.score,
                route_taken=final_route,
                verification_status=verdict.status,
                evidence_citations=output.citations,
                tools_used=[],
                uncertainty_reason=verdict.critique if verdict.status != VerificationStatus.PASS else None,
                telemetry=summary,
            )

        elif decision.path == RoutePath.TOOL:
            can_invoke, tool_err = self.state_manager.can_invoke_tool(state)
            if not can_invoke:
                status = VerificationStatus.ABSTAIN
                summary = telemetry.build_summary(state, status, tools_used=[], retrieval_used=False)
                if self.telemetry_log_file:
                    telemetry.append_to_jsonl(summary, self.telemetry_log_file)
                return AnswerResult(
                    answer="Tool execution limit reached under current safety policy.",
                    confidence=0.0,
                    route_taken=RoutePath.TOOL,
                    verification_status=status,
                    uncertainty_reason=tool_err,
                    telemetry=summary,
                )

            output = self.tool_path.execute(query, state, self.runtime)

            # Run Verifiers on tool output
            verdict = self.verifier.verify(
                query=query,
                candidate_answer=output.text,
                evidence=evidence or output.evidence,
                tool_results=output.tool_results,
            )
            telemetry.record_verification(verdict.status, score=verdict.score, critique=verdict.critique)

            if verdict.status == VerificationStatus.REVISE:
                can_retry, r_err = self.state_manager.can_retry(state)
                if can_retry:
                    self.state_manager.record_retry(state)
                    output = self.tool_path.execute(query, state, self.runtime)
                    verdict = self.verifier.verify(
                        query=query,
                        candidate_answer=output.text,
                        evidence=evidence or output.evidence,
                        tool_results=output.tool_results,
                    )

                if verdict.status == VerificationStatus.REVISE:
                    should_esc, target, rsn = self.state_manager.should_escalate(state, verdict.status, verdict.critique)
                    if target == RoutePath.ABSTAIN:
                        tools_used = [r.get("tool") for r in output.tool_results]
                        summary = telemetry.build_summary(state, VerificationStatus.ABSTAIN, tools_used=tools_used)
                        if self.telemetry_log_file:
                            telemetry.append_to_jsonl(summary, self.telemetry_log_file)
                        return AnswerResult(
                            answer="I am uncertain about this tool execution result as it failed verification.",
                            confidence=0.0,
                            route_taken=RoutePath.TOOL,
                            verification_status=VerificationStatus.ABSTAIN,
                            uncertainty_reason=verdict.critique or rsn,
                            telemetry=summary,
                        )

            tools_used = [r.get("tool") for r in output.tool_results]
            summary = telemetry.build_summary(state, verdict.status, tools_used=tools_used, retrieval_used=False)
            if self.telemetry_log_file:
                telemetry.append_to_jsonl(summary, self.telemetry_log_file)

            return AnswerResult(
                answer=output.text,
                confidence=verdict.score,
                route_taken=RoutePath.TOOL,
                verification_status=verdict.status,
                evidence_citations=output.citations,
                tools_used=tools_used,
                uncertainty_reason=verdict.critique if verdict.status != VerificationStatus.PASS else None,
                telemetry=summary,
            )

        elif decision.path == RoutePath.RETRIEVE:
            output = self.retrieve_path.execute(query, state, self.runtime)

            if not output.metadata.get("evidence_found", False):
                status = VerificationStatus.ABSTAIN
                telemetry.record_verification(
                    status,
                    score=0.0,
                    critique="Retrieval returned 0 relevant documents; abstaining under truthfulness policy."
                )
                summary = telemetry.build_summary(state, status, tools_used=[], retrieval_used=True)
                if self.telemetry_log_file:
                    telemetry.append_to_jsonl(summary, self.telemetry_log_file)

                return AnswerResult(
                    answer="I do not have verifiable external evidence to answer this inquiry accurately.",
                    confidence=0.0,
                    route_taken=RoutePath.RETRIEVE,
                    verification_status=status,
                    evidence_citations=[],
                    tools_used=[],
                    uncertainty_reason="No relevant grounding documents found in knowledge base.",
                    telemetry=summary,
                )

            verdict = self.verifier.verify(
                query=query,
                candidate_answer=output.text,
                evidence=output.evidence,
            )
            telemetry.record_verification(verdict.status, score=verdict.score, critique=verdict.critique)

            summary = telemetry.build_summary(state, verdict.status, tools_used=[], retrieval_used=True)
            if self.telemetry_log_file:
                telemetry.append_to_jsonl(summary, self.telemetry_log_file)

            return AnswerResult(
                answer=output.text,
                confidence=verdict.score,
                route_taken=RoutePath.RETRIEVE,
                verification_status=verdict.status,
                evidence_citations=output.citations,
                tools_used=[],
                uncertainty_reason=verdict.critique if verdict.status != VerificationStatus.PASS else None,
                telemetry=summary,
            )

        elif decision.path == RoutePath.THINK:
            output = self.think_path.execute(query, state, self.runtime)

            # Final verification of synthesized result
            verdict = self.verifier.verify(query=query, candidate_answer=output.text, evidence=evidence)
            telemetry.record_verification(verdict.status, score=verdict.score, critique=verdict.critique)

            if verdict.status == VerificationStatus.REVISE:
                status = VerificationStatus.ABSTAIN
                summary = telemetry.build_summary(state, status, tools_used=[], retrieval_used=False)
                if self.telemetry_log_file:
                    telemetry.append_to_jsonl(summary, self.telemetry_log_file)
                return AnswerResult(
                    answer="I am uncertain about the reasoning derivation for this complex query.",
                    confidence=0.0,
                    route_taken=RoutePath.THINK,
                    verification_status=status,
                    uncertainty_reason=verdict.critique or "Reasoning failed verification constraints.",
                    telemetry=summary,
                )

            summary = telemetry.build_summary(state, verdict.status, tools_used=[], retrieval_used=False)
            if self.telemetry_log_file:
                telemetry.append_to_jsonl(summary, self.telemetry_log_file)

            return AnswerResult(
                answer=output.text,
                confidence=verdict.score,
                route_taken=RoutePath.THINK,
                verification_status=verdict.status,
                evidence_citations=output.citations,
                tools_used=[],
                uncertainty_reason=verdict.critique if verdict.status != VerificationStatus.PASS else None,
                telemetry=summary,
            )

        # Default fallback
        output = self.fast_path.execute(query, state, self.runtime)
        status = VerificationStatus.UNVERIFIED
        telemetry.record_verification(status, score=output.confidence_estimate, critique="Executed via default fallback.")
        summary = telemetry.build_summary(state, status, tools_used=[], retrieval_used=False)

        return AnswerResult(
            answer=output.text,
            confidence=output.confidence_estimate * 0.7,
            route_taken=decision.path,
            verification_status=status,
            evidence_citations=[],
            tools_used=[],
            uncertainty_reason="Execution completed without formal verification step.",
            telemetry=summary,
        )
