# AI Group Project 2026 — RAG Pipeline for Mental Health Risk Detection

## Project Structure

AI-group-project-2026/
├── datasets/                           # Dataset and ChromaDB storage
│   ├── Suicide_Detection.csv           # Raw suicide dataset
│   ├── suicide_train.csv               # Training set (177,672 samples)
│   ├── suicide_test.csv                # Test set (44,419 samples)
│   ├── prepare_data.py                 # Dataset preparation script
│   └── chroma_db/                      # Training examples ChromaDB (auto-built)
│       └── train_suicide/              # 177,672 training embeddings
├── knowledge_based/                    # Knowledge base for clinical criteria
│   ├── mhgap.json                      # WHO mhGAP clinical criteria
│   └── kb_chromadb/                    # Knowledge base ChromaDB (auto-built)
│       └── knowledge_base/             # KB embeddings (3 chunks)
├── experiments/                        # Experiment runners
│   └── run_suicide.py                  # Main experiment runner
├── src/                                # Core source code
│   ├── config.py                       # Shared configuration
│   ├── embedder.py                     # SentenceTransformer wrapper
│   ├── llm_inference.py                # Ollama Qwen3 wrapper with thinking trace
│   ├── prompt_builder.py               # Prompt templates
│   ├── evaluate.py                     # Experiment runner with debug output
│   ├── vector_store.py                 # ChromaDB wrapper
│   └── retrievers/                     # KB retrieval implementations
│       ├── base.py                     # Abstract base class
│       ├── bm25_retriever.py           # Sparse keyword retrieval on KB
│       ├── dense_retriever.py          # Dense semantic retrieval on KB
│       └── hybrid_retriever.py         # BM25 + dense fusion on KB
├── results/                            # Experiment results (auto-generated)
├── data/                               # Source data (not committed)
│   ├── ICD-11-CDDR.pdf
│   ├── WHO-mhGAP-intervention-guide.pdf
│   └── depression.txt
├── notebooks/
│   └── analysis.ipynb                  # Analysis notebook
├── requirements.txt
└── README.md

All paths are resolved from the repo root via src/config.py.

## Setup

1. Clone the repo
   git clone https://github.com/Side941/AI-group-project-2026.git
   cd AI-group-project-2026

2. Install dependencies
   pip install -r requirements.txt

3. Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen3:0.6b
   ollama pull qwen3:1.7b
   ollama serve

4. Prepare datasets
   python3 datasets/prepare_data.py

Note: ChromaDB collections are auto-built on first run. No manual indexing needed.

## Running Experiments

Quick test: 1 sample, 1 experiment (zero-shot)
   python3 experiments/run_suicide.py --samples 1 --limit 1 --debug

KB retrieval: 1 sample, 2 experiments (zero-shot + KB)
   python3 experiments/run_suicide.py --samples 1 --limit 2 --debug

Combined mode (KB + Reddit examples): 1 sample, 10 experiments
   python3 experiments/run_suicide.py --samples 1 --limit 10 --debug

Full run with 10 samples
   python3 experiments/run_suicide.py --samples 10

Run all experiments with 100 samples
   python3 experiments/run_suicide.py --samples 100

Arguments:

   --samples   Number of test samples (default: 10)
   --limit     Number of experiments (default: All)
   --debug     Show debug output (default: False)

## Path Configuration (src/config.py)

   PROJECT_ROOT         Repo root
   SUICIDE_TRAIN_PATH   datasets/suicide_train.csv
   SUICIDE_TEST_PATH    datasets/suicide_test.csv
   MHGAP_SUICIDE_PATH   knowledge_based/mhgap.json
   CHROMA_PATH          datasets/chroma_db
   KB_CHROMA_PATH       knowledge_based/kb_chromadb
   LLM_MODEL_SIZES      ["0.6b", "1.7b"]
   KB_COLLECTION_NAME   "knowledge_base"

## Results

Saved to results/:

   summary_suicide_*.csv    All experiment summaries
   suicide_*_results.csv    Detailed predictions per experiment

Example Debug Output:

   🧠 Model Thinking/Reasoning:
   The clinical criteria mention suicide risk based on thoughts or plans...
   First, I need to check if the text contains any signs of imminent suicide risk...
   Therefore, the text doesn't indicate a suicide risk as per the criteria provided.

   📚 Retrieved KB Chunks (3):
     [1] Section: Assessment - Imminent Risk
         Text: Imminent risk of self-harm/suicide is indicated by...

## Notes

   - ChromaDB auto-builds on first run
   - Results are cached for faster re-runs
   - Default Top-K is 3, adjustable per experiment
   - Runs on CPU/MPS/GPU
   - Debug mode shows model thinking trace (🧠) for reasoning transparency
   - KB retrieval uses mhGAP clinical criteria from WHO
