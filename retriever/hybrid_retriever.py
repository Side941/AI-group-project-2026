# hybrid_retriever.py
from bm25_retriever import BM25Retriever
import components.ingestion as ingestion

class HybridRetriever:
    def __init__(self, chunks=None, json_path=None, alpha=0.3):
        self.bm25 = BM25Retriever(chunks=chunks, json_path=json_path)
        # DO NOT call initialise_retrieval() here
        # It is called once in build_retrievers() with the correct chroma_path
        self.alpha = alpha

    def search(self, query, k=5):
        # BM25 results
        bm25_results = self.bm25.search(query, k=k*2)

        # Dense results from ChromaDB
        query_embedding = ingestion._embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        raw = ingestion._collection.query(
            query_embeddings=query_embedding,
            n_results=k*2,
            include=["documents", "metadatas", "distances"],
        )

        dense_results = []
        for i in range(len(raw["documents"][0])):
            meta = raw["metadatas"][0][i]
            dense_results.append({
                "id":          f"{meta['disorder_code']}_{meta['section'].lower().replace(' ', '_')}",
                "text":        raw["documents"][0][i],
                "prompt_text": raw["documents"][0][i],
                "dense_score": 1 - raw["distances"][0][i],
                **meta
            })

        # Normalise scores
        bm25_max  = max(r['bm25_score']  for r in bm25_results)  if bm25_results  else 1
        dense_max = max(r['dense_score'] for r in dense_results) if dense_results else 1

        # Combine by chunk id
        combined = {}
        for r in bm25_results:
            combined[r['id']] = {
                'chunk':      r,
                'bm25_norm':  r['bm25_score'] / bm25_max if bm25_max > 0 else 0,
                'dense_norm': 0
            }
        for r in dense_results:
            dense_norm = r['dense_score'] / dense_max if dense_max > 0 else 0
            if r['id'] in combined:
                combined[r['id']]['dense_norm'] = dense_norm
            else:
                combined[r['id']] = {
                    'chunk':      r,
                    'bm25_norm':  0,
                    'dense_norm': dense_norm
                }

        # Score and rank
        scored = []
        for chunk_id, data in combined.items():
            hybrid_score = (self.alpha * data['bm25_norm'] +
                           (1 - self.alpha) * data['dense_norm'])
            chunk = data['chunk'].copy()
            chunk['hybrid_score'] = hybrid_score
            chunk['bm25_norm']    = data['bm25_norm']
            chunk['dense_norm']   = data['dense_norm']
            scored.append(chunk)

        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return scored[:k]