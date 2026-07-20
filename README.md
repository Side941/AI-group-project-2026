# AI Group Project 2026 — Depression / Suicide-Risk RAG Pipeline

Prompted RAG classifiers over ICD-11 clinical criteria (Qwen via Ollama). Two evaluation heads share the same retrieval stack:

| Head | Labels | Eval set | Notebook |
|------|--------|----------|----------|
| Multiclass | `suicidal` / `depression` / `normal` | 450 (150/class) | `notebooks/multi_class_rag.ipynb` |
| Binary | `suicide` / `non-suicide` | 300 (150/class) | `notebooks/binary_suicide_rag.ipynb` |

Each head also has a small **dev** slice (10/class) carved from its final set for prompt / top-k tuning.

## Project structure

```
AI-group-project-2026/
├── src/
│   ├── components/          # KB build: config, chunker, ingestion
│   └── retriever/           # BM25, dense, hybrid (+ section expander)
├── experiments/             # CLI runners
│   ├── run_kb_pipeline.py
│   ├── build_rag_eval_subset.py
│   └── build_binary_suicide_subset.py
├── notebooks/
│   ├── 01_kb_pipeline_demo.ipynb
│   ├── 02_dataset_prep_demo.ipynb
│   ├── multi_class_rag.ipynb
│   └── binary_suicide_rag.ipynb
├── datasets/                # CSVs only (source + eval/dev subsets)
├── results/                 # Meta JSON + experiment outputs
├── knowledge_base/          # ICD-11 PDF, chunks JSON, chroma_db/
├── rag_system_blueprint.html
├── requirements.txt
└── README.md
```

All paths resolve from the **repo root** via `src/components/config.py`. Experiment scripts and notebooks put `src/` (and `src/retriever/`) on `sys.path`.

## Setup

```bash
git clone https://github.com/Side941/AI-group-project-2026.git
cd AI-group-project-2026
pip install -r requirements.txt
```

**Also needed locally (not all committed):**

| Path | Role |
|------|------|
| `knowledge_base/icd_11.pdf` | Source PDF for chunking |
| `knowledge_base/icd11_chunks.json` | Chunked ICD-11 criteria |
| `knowledge_base/chroma_db/` | Dense vector store (gitignored) |
| Ollama + `qwen3:0.6b` / `qwen3:1.7b` | LLM inference for notebooks |

Committed eval/dev CSVs under `datasets/` and their provenance under `results/` are enough to run classification notebooks once the KB + Ollama are available.

## Evaluation sets

### Multiclass (mental-health corpus)

| File | Size | Purpose |
|------|------|---------|
| `datasets/rag_dev_slice.csv` | 30 (10/class) | Tuning (`eval_mode="dev"`) |
| `datasets/rag_eval_subset.csv` | 450 (150/class) | Reporting (`eval_mode="final"`) |

Anxiety excluded. Seeds: eval `42`, dev `43`. Meta: `results/rag_*.meta.json`.

```bash
python experiments/build_rag_eval_subset.py
```

Builder prefers local HF caches (`mental_heath_unbanlanced.csv`, `mental_health_combined_test.csv`); otherwise downloads from Hugging Face (`ourafla/Mental-Health_Text-Classification_Dataset`) and caches them.

### Binary suicide (`Suicide_Detection.csv`)

| File | Size | Purpose |
|------|------|---------|
| `datasets/binary_suicide_dev.csv` | 20 (10/class) | Tuning |
| `datasets/binary_suicide_eval.csv` | 300 (150/class) | Reporting |

Labels kept as in source: `suicide` / `non-suicide`. Meta: `results/binary_suicide_*.meta.json`.

```bash
python experiments/build_binary_suicide_subset.py
```

Requires `datasets/Suicide_Detection.csv` (large source; typically gitignored).

In either notebook, switch modes with `CFG["eval_mode"]` (`"dev"` or `"final"`). Default is `"dev"`. Lock prompts before switching to `"final"`.

## Running

**Knowledge-base pipeline** (from repo root):

```bash
python experiments/run_kb_pipeline.py
python experiments/run_kb_pipeline.py --skip-chunking   # ingest only
python experiments/run_kb_pipeline.py --rebuild         # recreate Chroma collection
```

Documented walkthrough: `notebooks/01_kb_pipeline_demo.ipynb`.

**Classification experiments** — open the notebook and run the path-setup cell first (auto-detects repo root):

- Multiclass → `notebooks/multi_class_rag.ipynb`
- Binary → `notebooks/binary_suicide_rag.ipynb`

Outputs land under `results/` (`rag_results_*.csv`, `binary_rag_results_*.csv`, error analyses).

**Dataset prep walkthrough:** `notebooks/02_dataset_prep_demo.ipynb` (multiclass builder stages).

## Path configuration

Edit once in `src/components/config.py`:

| Constant | Points to |
|----------|-----------|
| `PROJECT_ROOT` | Repo root (auto) |
| `PDF_PATH` / `CHUNKS_PATH` / `CHROMA_PATH` | `knowledge_base/` |
| `RAG_*` / `BINARY_*` | Eval/dev CSVs, meta, result paths |
| `RETRIEVAL_SECTIONS` / `MOOD_DISORDER_PREFIXES` | Retrieval filters over ICD-11 |

## Notes

- Retrievers: `bm25`, `dense`, `hybrid` (weighted RRF). Default top-k in experiments is configurable in the notebook `CFG`.
- Both heads currently retrieve over the **ICD-11** store (mood-filtered). Suicide-specific KB and few-shot vector store are planned (see `rag_system_blueprint.html`), not required to run today’s notebooks.
- CPU by default; GPU used when available for embeddings.
