"""
experiments/run_suicide.py
==========================
Run suicide detection experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from src.config import (
    ExperimentConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    HybridRetrieverConfig,
    SUICIDE_TRAIN_PATH,
    SUICIDE_TEST_PATH,
    RESULTS_DIR,
    LLM_MODEL_SIZES,
    THINKING_MODES,
)
from src.embedder import Embedder
from src.evaluate import run_experiment, compute_metrics


def generate_experiment_grid() -> List[ExperimentConfig]:
    """Generate all experiment configurations."""
    configs = []
    
    # Retriever configurations
    retrievers = [
        DenseRetrieverConfig(k=3, use_knowledge_base=False),
        DenseRetrieverConfig(k=5, use_knowledge_base=False),
        BM25RetrieverConfig(k=3),
        BM25RetrieverConfig(k=5),
        HybridRetrieverConfig(k=3, dense_weight=0.5),
        HybridRetrieverConfig(k=5, dense_weight=0.5),
        # Knowledge-based retrievers
        DenseRetrieverConfig(k=3, use_knowledge_base=True),
        DenseRetrieverConfig(k=5, use_knowledge_base=True),
    ]
    
    # For each model size
    for model_size in LLM_MODEL_SIZES:
        # For each thinking mode
        for thinking_mode in THINKING_MODES:
            # Zero-shot
            configs.append(ExperimentConfig(
                model_size=model_size,
                prompt_type="zero-shot",
                thinking_mode=thinking_mode,
                retriever=DenseRetrieverConfig(k=3, use_knowledge_base=False),
            ))
            
            # Few-shot with each retriever
            for retriever in retrievers:
                configs.append(ExperimentConfig(
                    model_size=model_size,
                    prompt_type="few-shot",
                    thinking_mode=thinking_mode,
                    retriever=retriever,
                ))
    
    return configs


def run_single_experiment(config: ExperimentConfig, test_df: pd.DataFrame, debug: bool = False) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    print(f"\n{'='*80}")
    print(f"Suicide | {config.model_size} | {config.prompt_type} | "
          f"{config.retriever.type} | k={config.retriever.k} | "
          f"KB={getattr(config.retriever, 'use_knowledge_base', False)} | "
          f"thinking={config.thinking_mode}")
    print(f"{'='*80}")
    
    # Load training data
    train_df = pd.read_csv(SUICIDE_TRAIN_PATH)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Run experiment
    embedder = Embedder()
    results_df = run_experiment(
        config=config,
        train_df=train_df,
        test_df=test_df,
        embedder=embedder,
        debug=debug
    )
    
    # Compute metrics
    y_true = results_df['true_label'].tolist()
    y_pred = results_df['pred_label'].tolist()
    metrics = compute_metrics(y_true, y_pred)
    
    # Add experiment metadata
    metrics.update({
        'model_size': config.model_size,
        'prompt_type': config.prompt_type,
        'thinking_mode': config.thinking_mode,
        'retriever_type': config.retriever.type,
        'retriever_k': config.retriever.k,
        'use_knowledge_base': getattr(config.retriever, 'use_knowledge_base', False),
        'timestamp': datetime.now().isoformat(),
        'test_size': len(test_df),
    })
    
    # Save detailed results
    config_name = f"suicide_{config.model_size}_{config.prompt_type}_{config.retriever.type}_k{config.retriever.k}"
    if getattr(config.retriever, 'use_knowledge_base', False):
        config_name += "_kb"
    if config.thinking_mode:
        config_name += "_think"
    
    results_path = RESULTS_DIR / f"{config_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to: {results_path}")
    
    return metrics


def run_all_experiments(debug: bool = False, limit: int = None, samples: int = None):
    """Run all suicide experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_df = pd.read_csv(SUICIDE_TEST_PATH)
    
    # Sample if specified
    if samples and samples < len(test_df):
        test_df = test_df.sample(n=samples, random_state=42)
        print(f"📊 Using {samples} random samples from test set (total: {len(pd.read_csv(SUICIDE_TEST_PATH)):,})")
    
    # Generate experiment grid
    configs = generate_experiment_grid()
    
    if limit:
        configs = configs[:limit]
    
    print(f"\n🚀 Running {len(configs)} suicide experiments")
    
    # Run experiments
    all_metrics = []
    for i, config in enumerate(configs):
        print(f"\n📊 Experiment {i+1}/{len(configs)}")
        try:
            metrics = run_single_experiment(config, test_df, debug=debug)
            all_metrics.append(metrics)
            print(f"✅ Accuracy: {metrics['accuracy']:.4f} | Nulls: {metrics['null_predictions']}")
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = RESULTS_DIR / f"summary_suicide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 Summary saved to: {summary_path}")
        
        print(f"\n🏆 Best accuracies:")
        best = summary_df.loc[summary_df['accuracy'].idxmax()]
        print(f"   {best['accuracy']:.4f} - {best['model_size']} | {best['prompt_type']} | "
              f"{best['retriever_type']} | KB={best['use_knowledge_base']}")
        
        print(f"\n🏆 Top 5 accuracies:")
        top5 = summary_df.nlargest(5, 'accuracy')[['model_size', 'prompt_type', 'retriever_type', 
                                                   'use_knowledge_base', 'accuracy', 'null_predictions']]
        print(top5.to_string(index=False))
    
    return all_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run suicide detection experiments')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of experiments')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples (default: 10)')
    
    args = parser.parse_args()
    
    print("Loading embedding model...")
    embedder = Embedder()
    _ = embedder.model
    
    run_all_experiments(debug=args.debug, limit=args.limit, samples=args.samples)


if __name__ == "__main__":
    main()