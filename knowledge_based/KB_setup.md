# ICD-11 Knowledge Base — Setup & Run Guide

## Project Structure

Place all five files in the same directory, alongside your `data/` folder:

```
your_project/
├── icd11_kb/
│   ├── config.py
│   ├── chunker.py
│   ├── ingestion.py
│   ├── retrieval.py
│   └── main.py
└── data/
    └── icd_11.pdf          ← your ICD-11 CDDR PDF goes here
```

---

## 1. Prerequisites

### Python
Python 3.10 or later is required (uses `str | None` union syntax).

### Conda environment (recommended)

```bash
conda create -n icd11_rag python=3.11
conda activate icd11_rag
```

### System dependency — pdftotext

`pdftotext` is used to extract text from the PDF. Install via conda:

```bash
conda install -c conda-forge poppler
```

Or on Ubuntu/Debian:

```bash
sudo apt install poppler-utils
```

---

## 2. Install Python Dependencies

```bash
pip install sentence-transformers chromadb tqdm torch
```

> **GPU acceleration (optional but recommended for ingestion speed):**
> If you have a CUDA-capable GPU, install the matching PyTorch build from
> https://pytorch.org/get-started/locally — the pipeline auto-detects CUDA.

---

## 3. Configure Paths

Open `config.py` and confirm these match your directory layout:

```python
PDF_PATH    = "../data/icd_11.pdf"     # path to ICD-11 CDDR PDF
CHUNKS_PATH = "../data/icd11_chunks.json"   # where chunks will be saved
CHROMA_PATH = "../data/chroma_db"           # where ChromaDB will be stored
```

Paths are relative to where you run the script from. If you run from inside
`icd11_kb/`, the defaults point one level up to `data/` — adjust as needed.

---

## 4. Run the Full Pipeline

From inside the `icd11_kb/` directory:

```bash
cd icd11_kb
python main.py
```

This runs all three stages in sequence:

| Stage | What happens |
|---|---|
| **Step 1 — Chunking** | Extracts pages 70–852 from the PDF, parses into ~2,000+ structured chunks, saves to `icd11_chunks.json` |
| **Step 2 — Ingestion** | Loads BioLORD-2023, embeds every chunk, stores vectors in ChromaDB |
| **Step 3 — Retrieval test** | Runs 5 sanity-check queries and prints results to stdout |

Expected runtime: ~5–15 min on CPU; ~1–3 min on GPU (ingestion dominates).

---

## 5. Run Individual Stages

You can run each stage independently — useful when iterating:

### Chunking only
```bash
python -c "from chunker import run_chunking; run_chunking()"
```

### Ingestion only (requires chunks JSON to exist)
```bash
python -c "from ingestion import run_ingestion; run_ingestion()"
```

### Force re-ingestion (wipe and rebuild the ChromaDB collection)
```bash
python -c "from ingestion import run_ingestion; run_ingestion(rebuild=True)"
```

---

## 6. Use in Your RAG Pipeline

Import the retrieval functions directly into your evaluation scripts:

```python
from retrieval import initialise_retrieval, get_icd11_context, query_icd11

# Call once at startup (skips re-embedding — just loads model + collection)
initialise_retrieval()

# Get a formatted context block for an LLM prompt
context = get_icd11_context(
    "I haven't slept in days and I feel like I'm falling apart",
    n_results=5,
)

# Inject into your prompt
prompt = f"""You are a clinical assistant helping assess mental health risk.
Use the ICD-11 clinical reference below to inform your response.

--- ICD-11 Clinical Reference ---
{context}
---------------------------------

Post: I haven't slept in days and I feel like I'm falling apart

Based on the clinical reference, what disorder categories are most relevant?"""
```

### Filter by section type

```python
# Only retrieve 'Essential Features' chunks
context = get_icd11_context(
    query_text="I can't stop checking the locks",
    n_results=3,
    section_filter="Essential Features",
)
```

Available section filters:

- `"Essential Features"`
- `"Additional Clinical Features"`
- `"Boundary with Normality"`
- `"Course Features"`
- `"Developmental Presentations"`
- `"Culture-Related Features"`
- `"Sex- and/or Gender-Related Features"`
- `"Differential Diagnosis"`
- `"Diagnostic Requirements"`
- `"Specifiers"`

---

## 7. Troubleshooting

### `pdftotext not found`
Run `conda install -c conda-forge poppler` and make sure the correct conda
environment is active.

### `RuntimeError: Retrieval not initialised`
You called `query_icd11()` or `get_icd11_context()` without first calling
either `run_ingestion()` or `initialise_retrieval()`.

### ChromaDB collection already exists
Re-running ingestion skips by default to avoid duplicates. To rebuild:
```python
run_ingestion(rebuild=True)
```

### Chunks JSON not found during ingestion
Run the chunking step first, or confirm `CHUNKS_PATH` in `config.py` points
to the correct location.

### Out of memory during embedding
Reduce `BATCH_SIZE` in `config.py` (default is 64). Try 16 or 32 on limited
hardware.
