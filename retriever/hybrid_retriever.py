# hybrid_retriever.py

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever

# HybridRetriever class that combines BM25 and dense retrieval methods
class HybridRetriever:
    def __init__(self, chunks=None, json_path=None, 
             alpha=0.5, model_name='all-MiniLM-L6-v2'):
        # Build both retrievers from the same chunks
        self.bm25 = BM25Retriever(chunks=chunks, json_path=json_path)
        self.dense = DenseRetriever(chunks=chunks, json_path=json_path, model_name=model_name)
        self.alpha = alpha  # weight: alpha*BM25 + (1-alpha)*dense
        
    def search(self, query, k=5):
        """Combine BM25 and dense scores, return re-ranked top-k."""
        # Get more candidates from each (k*2) to allow re-ranking
        bm25_results = self.bm25.search(query, k=k*2)
        dense_results = self.dense.search(query, k=k*2)
        
        # Normalize scores to 0-1 range
        bm25_max = max(r['bm25_score'] for r in bm25_results) if bm25_results else 1
        dense_max = max(r['dense_score'] for r in dense_results) if dense_results else 1
        
        # Combine scores by chunk_id
        combined = {}
        for r in bm25_results:
            combined[r['id']] = {
                'chunk': r,
                'bm25_norm': r['bm25_score'] / bm25_max if bm25_max > 0 else 0,
                'dense_norm': 0
            }
        
        # Add dense scores to the combined dict
        for r in dense_results:
            dense_norm = r['dense_score'] / dense_max if dense_max > 0 else 0
            if r['id'] in combined:
                combined[r['id']]['dense_norm'] = dense_norm
            else:
                combined[r['id']] = {
                    'chunk': r,
                    'bm25_norm': 0,
                    'dense_norm': dense_norm
                }
        
        # Calculate hybrid scores
        scored = []
        for chunk_id, data in combined.items():
            hybrid_score = (self.alpha * data['bm25_norm'] + 
                           (1 - self.alpha) * data['dense_norm'])
            chunk = data['chunk'].copy()
            chunk['hybrid_score'] = hybrid_score
            chunk['bm25_norm'] = data['bm25_norm']
            chunk['dense_norm'] = data['dense_norm']
            scored.append(chunk)
        
        # Sort by hybrid score and return top-k
        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return scored[:k]

