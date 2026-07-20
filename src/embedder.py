"""
embedder.py
===========
Generates embeddings for dataset examples and queries.
Uses sentence-transformers for dense vector representations.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, BATCH_SIZE, EMBEDDING_DIM


class Embedder:
    """Thin wrapper around SentenceTransformer for encoding texts."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = BATCH_SIZE,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            normalize: If True, embeddings are normalized to unit length (good for cosine).
            batch_size: Batch size for encoding.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        return np.array(embeddings)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string. Returns 1D array of shape (embedding_dim,)."""
        return self.encode([text], normalize=normalize)[0]