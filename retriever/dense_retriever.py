# dense_retriever.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import load_chunks

# DenseRetriever class for searching text chunks using dense embeddings and FAISS
class DenseRetriever:
    def __init__(self, chunks=None, json_path=None, model_name='all-MiniLM-L6-v2'):
        # Load chunks from JSON if not provided
        if chunks is None:
            if json_path:
                self.chunks = load_chunks(json_path)
            else:
                self.chunks = load_chunks()
        else:
            self.chunks = chunks

        # Load the embedding model
        self.model = SentenceTransformer(model_name)

        # Get embedding dimension from the model itself rather than hardcoding
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Encode all chunk texts
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(chunk_texts, normalize_embeddings=True)

        # Build FAISS index for inner product similarity search
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"Dense retriever ready: {len(self.chunks)} chunks indexed ({self.dimension} dims)")

    # Search method to retrieve top-k relevant chunks for a given query
    def search(self, query, k=5):
        # Handle empty or whitespace-only queries
        if not query or not query.strip():
            return []

        # Cap k at the number of available chunks to avoid FAISS errors
        k = min(k, len(self.chunks))

        # Encode query and search FAISS index
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['dense_score'] = float(score)
                results.append(chunk)

        return results