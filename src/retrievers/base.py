"""
retrievers/base.py
==================
Abstract base class for all retrievers.
Ensures BM25, Dense, and Hybrid all expose the same interface.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseRetriever(ABC):
    """Abstract retriever for example-based RAG."""

    def __init__(self, corpus_texts: List[str], corpus_labels: List[str]):
        """
        Args:
            corpus_texts: List of training example texts.
            corpus_labels: List of corresponding labels.
        """
        self.corpus_texts = corpus_texts
        self.corpus_labels = corpus_labels
        self._is_indexed = False

    @abstractmethod
    def index(self) -> None:
        """Build/precompute any index needed for retrieval (e.g., embeddings, BM25 weights)."""
        self._is_indexed = True

    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """
        Retrieve the Top-K most relevant examples from the corpus.

        Args:
            query_text: The raw query string.
            query_embedding: Pre-computed dense embedding (optional, used by Dense/Hybrid).
            k: Number of examples to retrieve.

        Returns:
            List of (text, label) tuples, ordered by relevance (most relevant first).
        """
        pass

    def is_indexed(self) -> bool:
        """Check if the retriever has been indexed."""
        return self._is_indexed