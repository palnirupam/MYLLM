"""
myllm.intelligence.retrieval.context_builder — Grounded context formatting.
Preserves document provenance and prevents hallucinated citations.
"""

from typing import List
from myllm.intelligence.retrieval.base import BaseContextBuilder
from myllm.intelligence.schemas import Document


class StructuredContextBuilder(BaseContextBuilder):
    """
    Constructs an XML/Markdown delimited context prompt from retrieved documents,
    retaining explicit provenance and citation tags.
    """

    def build_context(self, documents: List[Document]) -> str:
        if not documents:
            return ""

        context_blocks = []
        for doc in documents:
            provenance = doc.provenance_uri or doc.source or "unknown"
            header = f"<evidence doc_id=\"{doc.doc_id}\" source=\"{doc.source}\" uri=\"{provenance}\">"
            body = doc.content.strip()
            footer = "</evidence>"
            context_blocks.append(f"{header}\n{body}\n{footer}")

        joined_blocks = "\n\n".join(context_blocks)

        grounding_prompt = (
            "Grounded Evidence Documents:\n"
            f"{joined_blocks}\n\n"
            "Instructions: Answer the question using ONLY the factual statements in the evidence above. "
            "Cite the relevant doc_id for each claim made."
        )

        return grounding_prompt
