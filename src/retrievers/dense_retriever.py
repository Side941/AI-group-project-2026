"""
src/retrievers/dense_retriever.py
==================================
Dense retriever for knowledge base (mhGAP chunks) using ChromaDB.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .base import BaseRetriever
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.config import KB_CHROMA_PATH, KB_COLLECTION_NAME


class DenseRetriever(BaseRetriever):
    """
    Dense retriever for knowledge base.
    """

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: List[str],
        embedder: Embedder | None = None,
    ):
        super().__init__(corpus_texts, corpus_labels)
        self.embedder = embedder or Embedder()
        self.vector_store = VectorStore(KB_CHROMA_PATH, KB_COLLECTION_NAME)
        self._is_indexed = False

    def index(self) -> None:
        """Index KB into ChromaDB."""
        self.vector_store.create_or_load(384)
        
        if self.vector_store.count() == 0:
            print("Indexing knowledge base into ChromaDB...")
            embeddings = self.embedder.encode(self.corpus_texts)
            ids = [f"kb_{i:04d}" for i in range(len(self.corpus_texts))]
            metadatas = [{'section': label} for label in self.corpus_labels]
            self.vector_store.add_with_metadata(ids, self.corpus_texts, embeddings, metadatas, self.corpus_labels)
            print(f"Indexed {self.vector_store.count()} KB chunks")
        else:
            print(f"Using cached KB: {self.vector_store.count()} chunks")
        
        self._is_indexed = True

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray | None = None,
        k: int = 3,
    ) -> List[Tuple[str, str]]:
        """Retrieve top-k KB chunks."""
        if not self._is_indexed:
            raise RuntimeError("DenseRetriever not indexed. Call index() first.")

        if query_embedding is None:
            query_embedding = self.embedder.encode_single(query_text)

        results = self.vector_store.query(query_embedding, k=k)
        return [(doc, meta.get('section', '')) for doc, meta in results]