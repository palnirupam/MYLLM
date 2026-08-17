"""
myllm.intelligence.retrieval.bm25 — In-memory BM25 retrieval baseline.
Provides fast, deterministic lexical search with language & metadata filtering.
"""

import math
import re
from typing import List, Optional, Dict, Any
from myllm.intelligence.retrieval.base import SparseRetriever
from myllm.intelligence.schemas import Document
import string
import unicodedata

_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation + "।॥—‘’“”«»`~@#$%^&*()_+=<>{}[]|/\\")


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into Unicode NFC-normalized words across English, Bengali, Hindi, and other scripts."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned = normalized.translate(_PUNCT_TRANSLATION)
    return [w.lower() for w in cleaned.split() if len(w) > 0]


class InMemoryBM25Retriever(SparseRetriever):
    """
    Standard BM25 lexical retriever with document-length normalization,
    IDF caching, language filtering, and metadata constraints.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            tokens = _tokenize(doc.content)
            self.documents.append(doc)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Update document frequency for unique tokens in this doc
            seen = set(tokens)
            for token in seen:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        total_docs = len(self.documents)
        if total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths) / total_docs
            # Precompute IDFs
            for token, df in self.doc_freqs.items():
                self.idf_cache[token] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        language: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Document]:
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: List[tuple[float, int]] = []

        for idx, (doc, tokens, doc_len) in enumerate(zip(self.documents, self.doc_tokens, self.doc_lengths)):
            # Language filter
            if language is not None and doc.language != language:
                continue

            # Metadata filter
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Compute BM25 score
            doc_score = 0.0
            if doc_len == 0:
                continue

            # Token frequencies in this document
            token_counts: Dict[str, int] = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            for q_tok in query_tokens:
                if q_tok in token_counts:
                    tf = token_counts[q_tok]
                    idf = self.idf_cache.get(q_tok, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_length)))
                    doc_score += idf * (numerator / max(1e-6, denominator))

            if doc_score >= min_score and doc_score > 0.0:
                scores.append((doc_score, idx))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for score, idx in scores[:top_k]:
            orig_doc = self.documents[idx]
            # Return new Document instance with populated score
            results.append(
                Document(
                    doc_id=orig_doc.doc_id,
                    content=orig_doc.content,
                    score=round(score, 4),
                    source=orig_doc.source,
                    provenance_uri=orig_doc.provenance_uri,
                    language=orig_doc.language,
                    timestamp=orig_doc.timestamp,
                    metadata=orig_doc.metadata,
                )
            )

        return results
