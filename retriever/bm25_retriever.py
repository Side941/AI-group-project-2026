# bm25_retriever.py

import numpy as np
from rank_bm25 import BM25Okapi
from utils import load_chunks, tokenize

# BM25Retriever class for searching text chunks using BM25 algorithm
class BM25Retriever:
    def __init__(self, chunks=None, json_path=None):
        # Load chunks from JSON if not provided
        if chunks is None:
            if json_path:
                self.chunks = load_chunks(json_path)
            else:
                self.chunks = load_chunks()
        else:
            self.chunks = chunks

        # Tokenize the text in each chunk for BM25 indexing
        # Use prompt_text (clean clinical text) not text (embed_text with metadata headers)
        # embed_text contains repeated generic words like "Source", "Domain", "Disorder"
        # which pollute BM25 keyword matching and cause false positives
        self.tokenized_chunks = []
        for chunk in self.chunks:
            self.tokenized_chunks.append(tokenize(chunk.get('prompt_text') or chunk['text']))

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"BM25 retriever ready: {len(self.chunks)} chunks indexed")

    # Search method to retrieve top-k relevant chunks for a given query
    def search(self, query, k=5):
        # Handle empty or whitespace-only queries
        if not query or not query.strip():
            return []
        tokenized_query = tokenize(query)

        # Handle case where tokenization results in an empty list
        if not tokenized_query:
            return []

        # Get BM25 scores for the tokenized query
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top-k scores sorted descending
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            # Only include chunks with a positive score
            if scores[idx] > 0:
                chunk = self.chunks[idx].copy()
                chunk['bm25_score'] = float(scores[idx])
                results.append(chunk)

        return results