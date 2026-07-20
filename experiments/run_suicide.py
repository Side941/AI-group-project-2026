"""
experiments/run_suicide.py
==========================
Run suicide detection experiments with KB retrieval (Dense, BM25, Hybrid).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Literal, List, Dict, Any

from src.config import (
    SUICIDE_TRAIN_PATH,
    SUICIDE_TEST_PATH,
    RESULTS_DIR,
    LLM_MODEL_SIZES,
    THINKING_MODES,
    get_labels,
    TEXT_COL,
    LABEL_COL,
)
from src.embedder import Embedder
from src.evaluate import run_experiment, compute_metrics


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    model_size: str
    prompt_type: Literal["zero-shot", "few-shot"]
    thinking_mode: bool
    retriever_type: Literal["dense", "bm25", "hybrid"]
    k: int = 3
    use_examples: bool = False  # If True, include Reddit examples with KB


def generate_experiment_grid() -> List[ExperimentConfig]:
    """Generate all experiment configurations."""
    configs = []
    retriever_types = ["dense", "bm25", "hybrid"]
    k_values = [3, 5, 7, 10]

    for model_size in LLM_MODEL_SIZES:
        for thinking_mode in THINKING_MODES:
            # Zero-shot
            configs.append(ExperimentConfig(
                model_size=model_size,
                prompt_type="zero-shot",
                thinking_mode=thinking_mode,
                retriever_type="dense",
                k=3,
                use_examples=False,
            ))

            # Few-shot with KB only (no Reddit examples)
            for retriever_type in retriever_types:
                for k in k_values:
                    configs.append(ExperimentConfig(
                        model_size=model_size,
                        prompt_type="few-shot",
                        thinking_mode=thinking_mode,
                        retriever_type=retriever_type,
                        k=k,
                        use_examples=False,
                    ))

            # Few-shot with KB + Reddit examples (combined)
            for retriever_type in retriever_types:
                for k in k_values:
                    configs.append(ExperimentConfig(
                        model_size=model_size,
                        prompt_type="few-shot",
                        thinking_mode=thinking_mode,
                        retriever_type=retriever_type,
                        k=k,
                        use_examples=True,
                    ))

    return configs


def run_single_experiment(config: ExperimentConfig, train_df: pd.DataFrame, test_df: pd.DataFrame, debug: bool = False) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    mode = "Combined (KB+Examples)" if config.use_examples else "KB only"
    print(f"\n{'='*80}")
    print(f"Suicide | {config.model_size} | {config.prompt_type} | "
          f"{config.retriever_type} | k={config.k} | {mode} | "
          f"thinking={config.thinking_mode}")
    print(f"{'='*80}")
    
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
        'retriever_type': config.retriever_type,
        'k': config.k,
        'use_examples': config.use_examples,
        'timestamp': datetime.now().isoformat(),
        'test_size': len(test_df),
    })
    
    # Save detailed results
    config_name = f"suicide_{config.model_size}_{config.prompt_type}_{config.retriever_type}_k{config.k}"
    if config.use_examples:
        config_name += "_combined"
    else:
        config_name += "_kbonly"
    if config.thinking_mode:
        config_name += "_think"
    
    results_path = RESULTS_DIR / f"{config_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to: {results_path}")
    
    return metrics


def run_all_experiments(debug: bool = False, limit: int = None, samples: int = None):
    """Run all suicide experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(SUICIDE_TRAIN_PATH)
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
    print(f"   - KB only: retrieves clinical criteria from mhGAP")
    print(f"   - Combined: retrieves clinical criteria + Reddit examples")
    
    # Run experiments
    all_metrics = []
    for i, config in enumerate(configs):
        print(f"\n📊 Experiment {i+1}/{len(configs)}")
        try:
            metrics = run_single_experiment(config, train_df, test_df, debug=debug)
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
        mode = "Combined" if best['use_examples'] else "KB only"
        print(f"   {best['accuracy']:.4f} - {best['model_size']} | {best['prompt_type']} | "
              f"{best['retriever_type']} | k={best['k']} | {mode}")
        
        print(f"\n🏆 Top 5 accuracies:")
        top5 = summary_df.nlargest(5, 'accuracy')[['model_size', 'prompt_type', 'retriever_type', 
                                                   'k', 'use_examples', 'accuracy', 'null_predictions']]
        print(top5.to_string(index=False))
    
    return all_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run suicide detection experiments with KB')
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