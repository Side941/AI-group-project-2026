# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

AI-group-project-2026/
├── datasets/ # Dataset and ChromaDB storage
│ ├── Suicide_Detection.csv # Raw suicide dataset
│ ├── suicide_train.csv # Training set (177,672 samples)
│ ├── suicide_test.csv # Test set (44,419 samples)
│ ├── prepare_data.py # Dataset preparation script
│ └── chroma_db/ # Example-based ChromaDB (auto-built)
│ └── train_suicide/ # 177,672 training embeddings
├── knowledge_based/ # Knowledge base for clinical criteria
│ ├── mhgap.json # WHO mhGAP clinical criteria
│ └── kb_chromadb/ # Knowledge base ChromaDB (auto-built)
│ └── knowledge_base/ # KB embeddings (3 chunks)
├── experiments/ # Experiment runners
│ ├── run_suicide.py # Run suicide detection experiments
│ └── run_suicide_kb.py # Run KB-only experiments
├── src/ # Core source code
│ ├── config.py # Shared configuration
│ ├── embedder.py # SentenceTransformer wrapper
│ ├── llm_inference.py # Ollama Qwen3 wrapper
│ ├── prompt_builder.py # Prompt templates
│ ├── evaluate.py # Experiment runner
│ ├── vector_store.py # ChromaDB wrapper
│ ├── section_expander.py # Section expansion for KB retrieval
│ └── retrievers/ # Retrieval implementations
│ ├── base.py # Abstract base class
│ ├── bm25_retriever.py # Sparse keyword retrieval
│ ├── dense_retriever.py # Dense semantic retrieval
│ └── hybrid_retriever.py # BM25 + dense fusion
├── results/ # Experiment results (auto-generated)
├── data/ # Source data (not committed)
│ ├── ICD-11-CDDR.pdf
│ ├── WHO-mhGAP-intervention-guide.pdf
│ └── depression.txt
├── notebooks/
│ └── analysis.ipynb # Analysis notebook
├── requirements.txt
└── README.md

# AI Group Project 2026 — RAG Pipeline for Suicide Risk Detection

## Project Structure

```
AI-group-project-2026/
├── datasets/
│   ├── Suicide_Detection.csv
│   ├── suicide_train.csv
│   ├── suicide_test.csv
│   ├── prepare_data.py
│   └── chroma_db/
│       └── train_suicide/
├── knowledge_based/
│   ├── mhgap.json
│   └── kb_chromadb/
│       └── knowledge_base/
├── experiments/
│   ├── run_suicide.py
│   └── run_suicide_kb.py
├── src/
│   ├── config.py
│   ├── embedder.py
│   ├── llm_inference.py
│   ├── prompt_builder.py
│   ├── evaluate.py
│   ├── vector_store.py
│   ├── section_expander.py
│   └── retrievers/
│       ├── base.py
│       ├── bm25_retriever.py
│       ├── dense_retriever.py
│       └── hybrid_retriever.py
├── results/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

All paths are resolved from the **repo root** via `src/config.py`.

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

**3. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama serve
```

**4. Prepare datasets**
```bash
python3 datasets/prepare_data.py
```

## Running Experiments

**Suicide detection:**
```bash
python3 experiments/run_suicide.py --samples 3 --limit 1 --debug
python3 experiments/run_suicide.py --samples 10 --limit 5 --debug
python3 experiments/run_suicide.py --samples 100
```

**Knowledge-base only:**
```bash
python3 experiments/run_suicide_kb.py --samples 10 --limit 5 --debug
python3 experiments/run_suicide_kb.py --samples 100
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--samples` | Number of test samples | 10 |
| `--limit` | Number of experiments | All |
| `--debug` | Show debug output | False |

## Path Configuration (`src/config.py`)

| Constant | Points to |
|----------|-----------|
| `PROJECT_ROOT` | Repo root |
| `SUICIDE_TRAIN_PATH` | `datasets/suicide_train.csv` |
| `SUICIDE_TEST_PATH` | `datasets/suicide_test.csv` |
| `MHGAP_SUICIDE_PATH` | `knowledge_based/mhgap.json` |
| `CHROMA_PATH` | `datasets/chroma_db` |
| `KB_CHROMA_PATH` | `knowledge_based/kb_chromadb` |
| `LLM_MODEL_SIZES` | `["0.6b", "1.7b"]` |
| `KB_COLLECTION_NAME` | `"knowledge_base"` |

## Results

Saved to `results/`:

| File | Description |
|------|-------------|
| `summary_suicide_*.csv` | All experiment summaries |
| `summary_suicide_kb_*.csv` | KB-only summaries |
| `suicide_*_results.csv` | Detailed predictions |
| `suicide_kb_*_results.csv` | KB predictions |

**Key Findings (10 samples):**

| Method | Best k | Accuracy |
|--------|--------|----------|
| Zero-shot | - | ~50-66% |
| Example-based Dense | 3/5 | 100% |
| Knowledge Base | 10 | 90% |

## Retriever Types

| Retriever | Description |
|-----------|-------------|
| **Dense** | Semantic similarity |
| **BM25** | Keyword matching |
| **Hybrid** | Dense + BM25 fusion |
| **KB** | Clinical criteria (mhGAP) |

## Notes

- ChromaDB auto-builds on first run
- Results are cached
- Default Top-K is 3
- Runs on CPU/MPS/GPU
