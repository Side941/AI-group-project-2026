# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

```
AI-group-project-2026/
├── components/                 # Knowledge-base build pipeline
│   ├── config.py               # Shared config + repo-root path resolution
│   ├── chunker.py              # Step 1: ICD-11 PDF → JSON chunks
│   └── ingestion.py            # Step 2: JSON chunks → ChromaDB vectors
├── retriever/                  # Retrieval layer
│   ├── paths.py                # Import-path bootstrap for scripts
│   ├── utils.py                # Load chunks, tokenize for BM25
│   ├── bm25_retriever.py       # Sparse keyword retrieval
│   ├── retrieval.py            # ChromaDB dense retrieval API
│   ├── retrieval_retriever.py  # Dense retriever + same-disorder expansion
│   ├── hybrid_retriever.py     # BM25 + dense fusion
│   └── test.py                 # Smoke test for all retrievers
├── notebooks/
│   └── multi_class_rag.ipynb   # Full RAG classification experiment
├── data/
│   └── icd_11.pdf              # Source ICD-11 PDF (not committed)
├── knowledge_based/
│   ├── icd11_chunks.json       # Extracted clinical chunks
│   └── chroma_db/              # Vector store (gitignored)
├── datasets/
│   └── Depression_Severity_Levels_Dataset.csv
├── requirements.txt
└── README.md
```

All paths are resolved from the **repo root** via `components/config.py`, so code works whether you run from the project root, `retriever/`, or `notebooks/`.

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

**3. Place required data files**
- `data/icd_11.pdf`
- `knowledge_based/icd11_chunks.json` (from chunker, or provided)
- `knowledge_based/chroma_db/` (from ingestion, or provided)
- `datasets/Depression_Severity_Levels_Dataset.csv`

## Running

**Retriever smoke test** (from anywhere):
```bash
python retriever/test.py
```

**Knowledge-base pipeline** (from project root):
```bash
python -m components.chunker
python -m components.ingestion
```

**Notebook experiment** — open `notebooks/multi_class_rag.ipynb` and run cell 1 first. It auto-detects the project root.

## Path configuration

Edit paths in one place: `components/config.py`

| Constant | Points to |
|----------|-----------|
| `PROJECT_ROOT` | Repo root (auto-detected) |
| `PDF_PATH` | `data/icd_11.pdf` |
| `CHUNKS_PATH` | `knowledge_based/icd11_chunks.json` |
| `CHROMA_PATH` | `knowledge_based/chroma_db` |
| `DATASET_PATH` | `datasets/Depression_Severity_Levels_Dataset.csv` |

## Notes
- Default Top-K is 5, adjustable via the `k` parameter in `search()`.
- Runs on CPU by default; uses GPU when available.
- Retriever types: `bm25`, `hybrid`, `retrieval`.
