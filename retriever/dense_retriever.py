# dense_retriever.py
import components.ingestion as ingestion

class DenseRetriever:
    def __init__(self):
        pass  # initialise_retrieval() already called in build_retrievers()

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        query_embedding = ingestion._embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        raw = ingestion._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for i in range(len(raw["documents"][0])):
            meta = raw["metadatas"][0][i]
            results.append({
                "id":          f"{meta['disorder_code']}_{meta['section'].lower().replace(' ', '_')}",
                "text":        raw["documents"][0][i],
                "prompt_text": raw["documents"][0][i],
                "dense_score": 1 - raw["distances"][0][i],
                **meta
            })
        return results