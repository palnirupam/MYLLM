"""
myllm.intelligence.paths.retrieve_path — Evidence-grounded retrieval path.
Coordinates document retrieval, context building, grounded generation, and provenance tracking.
"""

from typing import Optional, List, Dict, Any
from myllm.intelligence.paths.base import BasePath, PathOutput
from myllm.intelligence.schemas import ExecutionState, Document
from myllm.intelligence.retrieval.base import BaseRetriever, BaseContextBuilder
from myllm.intelligence.retrieval.bm25 import InMemoryBM25Retriever
from myllm.intelligence.retrieval.context_builder import StructuredContextBuilder
from myllm.runtime.interfaces.base import InferenceRuntime


class RetrievePath(BasePath):
    """
    Executes retrieval against registered knowledge documents, injects grounded context,
    and returns answer candidates tagged with verifiable evidence citations.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        context_builder: Optional[BaseContextBuilder] = None,
    ):
        self.retriever = retriever or InMemoryBM25Retriever()
        self.context_builder = context_builder or StructuredContextBuilder()

    def execute(
        self,
        query: str,
        state: ExecutionState,
        runtime: InferenceRuntime,
    ) -> PathOutput:
        # Retrieve top_k documents
        top_k = state.budget.max_tool_calls or 3
        retrieved_docs: List[Document] = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=0.1,
        )

        if not retrieved_docs:
            # Zero usable evidence found -> return empty evidence for verifier / abstention
            return PathOutput(
                text="No verifiable external evidence found for this inquiry.",
                tokens_generated=10,
                evidence=[],
                citations=[],
                confidence_estimate=0.0,
                metadata={
                    "path_type": "RETRIEVE",
                    "retrieved_count": 0,
                    "evidence_found": False,
                },
            )

        # Build grounded context
        grounded_context = self.context_builder.build_context(retrieved_docs)
        augmented_prompt = f"{grounded_context}\n\nUser Question: {query}\n\nAnswer:"

        # Generate answer with model
        generated_text = runtime.generate(
            prompt=augmented_prompt,
            max_new_tokens=state.budget.max_tokens,
            temperature=state.budget.temperature,
            top_k=state.budget.top_k,
            top_p=state.budget.top_p,
        )

        if generated_text.startswith(augmented_prompt):
            answer_text = generated_text[len(augmented_prompt):].strip()
        else:
            answer_text = generated_text.strip()

        est_tokens = max(1, len(answer_text.split()))
        state.tokens_consumed += est_tokens

        # Collect verified citations directly from retrieved documents (preventing fabricated citations)
        citations = [doc.doc_id for doc in retrieved_docs]
        evidence_contents = [f"[doc_id='{doc.doc_id}' source='{doc.source}'] {doc.content}" for doc in retrieved_docs]

        return PathOutput(
            text=answer_text,
            tokens_generated=est_tokens,
            evidence=evidence_contents,
            citations=citations,
            confidence_estimate=0.90,
            metadata={
                "path_type": "RETRIEVE",
                "retrieved_count": len(retrieved_docs),
                "selected_doc_ids": citations,
                "evidence_found": True,
            },
        )
