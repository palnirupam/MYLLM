"""
myllm.intelligence.retrieval.base — Provider-neutral retrieval abstractions and interfaces.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from myllm.intelligence.schemas import Document


class BaseRetriever(ABC):
    """
    Abstract contract for all evidence retrieval backends.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        language: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Document]:
        """
        Retrieves relevant documents matching the query and filtering criteria.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Indexes documents into the retriever.
        """
        pass


class SparseRetriever(BaseRetriever):
    """Abstract interface for keyword/lexical retrievers (BM25, TF-IDF)."""
    pass


class DenseRetriever(BaseRetriever):
    """Abstract interface for embedding/vector-based retrievers."""
    pass


class HybridRetriever(BaseRetriever):
    """Abstract interface combining sparse and dense retrieval strategies."""
    pass


class BaseReranker(ABC):
    """
    Abstract interface for cross-encoder or neural reranking models.
    """

    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        pass


class BaseContextBuilder(ABC):
    """
    Abstract interface for transforming retrieved documents into grounded context prompts.
    """

    @abstractmethod
    def build_context(self, documents: List[Document]) -> str:
        """
        Formats documents with strict provenance tags, preventing citation fabrication.
        """
        pass
