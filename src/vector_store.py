"""
src/vector_store.py
===================
Simple ChromaDB wrapper for storing and retrieving example embeddings.
"""
from __future__ import annotations

import chromadb
import numpy as np
from pathlib import Path
from typing import List, Tuple


class VectorStore:
    """Thin wrapper around ChromaDB for example-based retrieval."""

    def __init__(self, persist_path: str | Path, collection_name: str = "examples"):
        self.persist_path = str(persist_path)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=self.persist_path)
        self._collection = None

    def create_or_load(self, embedding_dim: int) -> None:
        """Create a new collection or load existing one."""
        existing = [c.name for c in self._client.list_collections()]
        if self.collection_name in existing:
            self._collection = self._client.get_collection(self.collection_name)
            print(f"Loaded collection '{self.collection_name}' with {self._collection.count()} vectors")
        else:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Created collection '{self.collection_name}'")

    def add(self, ids: List[str], texts: List[str], embeddings: np.ndarray, labels: List[str], batch_size: int = 5000) -> None:
        """Add examples to the collection in batches."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=[{"label": lbl} for lbl in labels[start:end]],
            )
            if end % 50000 == 0 or end == total:
                print(f"  Indexed {end:,}/{total:,} examples")

    def add_with_metadata(self, ids: List[str], texts: List[str], embeddings: np.ndarray, metadatas: List[dict], labels: List[str], batch_size: int = 5000) -> None:
        """Add examples with custom metadata to the collection in batches."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            # Add label to metadata for each item
            batch_metadatas = []
            for i in range(start, end):
                meta = metadatas[i].copy()
                meta['label'] = labels[i]
                batch_metadatas.append(meta)
            
            self._collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=batch_metadatas,
            )
            if end % 50000 == 0 or end == total:
                print(f"  Indexed {end:,}/{total:,} examples")

    def query(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, dict]]:
        """Retrieve top-k most similar examples with metadata."""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas"],
        )
        
        examples = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            examples.append((doc, meta))
        
        return examples

    def query_all(self, query_embedding: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Return ALL examples with their similarity scores.
        Used by HybridRetriever for dense score fusion.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self._collection.count(),
            include=["distances"],
        )
        
        ids = results["ids"][0]
        distances = np.array(results["distances"][0])
        similarities = 1.0 - distances
        
        return ids, similarities

    def count(self) -> int:
        return self._collection.count() if self._collection else 0