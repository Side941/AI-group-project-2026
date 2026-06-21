# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

```
AI-group-project-2026/
  knowledge_based/
    cddr_chunks.json        # Placeholder CDDR chunks (real chunks coming)
  retriever/
    utils.py                # Shared utilities (load chunks, tokenize)
    bm25_retriever.py       # BM25 sparse retrieval
    dense_retriever.py      # Dense retrieval using sentence-transformers + FAISS
    hybrid_retriever.py     # Hybrid BM25 + dense retrieval
    test.py                 # Test script for all three retrievers
  requirements.txt
  README.md
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Side941/AI-group-project-2026.git
cd AI-group-project-2026
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Retriever

Run from the project root:

```bash
python3 retriever/test.py
```

## Notes
- `cddr_chunks.json` contains placeholder chunks — real ICD-11 CDDR chunks will replace this file once extraction is complete, no code changes needed.
- Default Top-K is 5, adjustable via the `k` parameter in `search()`.
- Runs entirely on CPU, no GPU required.