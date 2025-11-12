"""
View and Compare Experiment Results from GCS
Utility to list, compare, and download results from Google Cloud Storage
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate
from gcs_manager import GCSManager


def list_experiments(gcs_manager: GCSManager) -> List[str]:
    """List all experiments in the bucket."""
    print("📦 Listing experiments from GCS...")
    experiments = gcs_manager.list_experiments()

    if not experiments:
        print("No experiments found in bucket.")
        return []

    print(f"\nFound {len(experiments)} experiments:\n")
    for i, exp_id in enumerate(experiments, 1):
        print(f"{i:3}. {exp_id}")

    return experiments


def load_all_metrics(
    gcs_manager: GCSManager,
    experiments: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Load metrics for all experiments."""
    all_metrics = {}

    print("\n📊 Loading metrics...")
    for exp_id in experiments:
        metrics = gcs_manager.load_experiment_metrics(exp_id)
        if metrics:
            all_metrics[exp_id] = metrics
        else:
            print(f"⚠️  No metrics found for {exp_id}")

    return all_metrics


def create_comparison_table(
    all_metrics: Dict[str, Dict[str, Any]],
    sort_by: str = 'best_reward'
) -> pd.DataFrame:
    """Create a comparison table from all metrics."""
    rows = []

    for exp_id, metrics in all_metrics.items():
        # Extract config if available
        config = metrics.get('hyperparameters', {})

        row = {
            'experiment_id': exp_id[:20] + '...' if len(exp_id) > 20 else exp_id,
            'best_reward': metrics.get('best_reward', 0),
            'final_mean_reward': metrics.get('final_mean_reward', 0),
            'final_max_reward': metrics.get('final_max_reward', 0),
            'best_score': metrics.get('best_score', 0),
            'final_mean_score': metrics.get('final_mean_score', 0),
            'total_episodes': metrics.get('total_episodes', 0),
            'training_time_min': metrics.get('training_time_minutes', 0),
            'steps_per_sec': metrics.get('steps_per_second', 0),
            'lr_start': config.get('lr_start', 0),
            'lr_schedule': config.get('lr_schedule', ''),
            'ent_start': config.get('ent_start', 0),
            'hidden_size': config.get('hidden_size', 0),
            'n_envs': config.get('n_envs', 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by specified column
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)

    return df


def display_table(df: pd.DataFrame, format: str = 'simple', max_rows: Optional[int] = None):
    """Display dataframe as formatted table."""
    if max_rows:
        df = df.head(max_rows)

    print("\n" + "=" * 120)
    print(tabulate(df, headers='keys', tablefmt=format, showindex=False, floatfmt='.2f'))
    print("=" * 120)


def save_comparison_csv(df: pd.DataFrame, output_path: str):
    """Save comparison table to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\n💾 Comparison table saved to: {output_path}")


def show_experiment_details(gcs_manager: GCSManager, experiment_id: str):
    """Show detailed information for a specific experiment."""
    print(f"\n{'='*80}")
    print(f"Experiment Details: {experiment_id}")
    print(f"{'='*80}\n")

    # Load config
    config = gcs_manager.load_config(experiment_id)
    if config:
        print("📋 Configuration:")
        print(json.dumps(config.get('hyperparameters', {}), indent=2))
    else:
        print("⚠️  Config not found")

    # Load metrics
    metrics = gcs_manager.load_experiment_metrics(experiment_id)
    if metrics:
        print("\n📊 Metrics:")
        print(json.dumps(metrics, indent=2))
    else:
        print("⚠️  Metrics not found")

    print(f"\n{'='*80}")


def download_best_checkpoint(
    gcs_manager: GCSManager,
    experiment_id: str,
    output_path: Optional[str] = None
):
    """Download the best checkpoint from an experiment."""
    if output_path is None:
        output_path = f"downloaded_{experiment_id}_best.pt"

    print(f"\n📥 Downloading best checkpoint from {experiment_id}...")

    success = gcs_manager.download_best_checkpoint(experiment_id, output_path)

    if success:
        print(f"✅ Checkpoint downloaded to: {output_path}")
    else:
        print(f"❌ Failed to download checkpoint")


def find_best_experiments(
    all_metrics: Dict[str, Dict[str, Any]],
    metric: str = 'best_reward',
    top_k: int = 5
) -> List[tuple]:
    """Find top-k experiments by specified metric."""
    scored = []

    for exp_id, metrics in all_metrics.items():
        score = metrics.get(metric, 0)
        scored.append((exp_id, score, metrics))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser(description="View and compare PPO Flappy Bird experiment results from GCS")

    parser.add_argument(
        '--bucket',
        type=str,
        default='ppo-flappy-bird',
        help='GCS bucket name'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='quiet-sum-477223-g3',
        help='GCP project ID'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all experiments'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all experiments in a table'
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        default='best_reward',
        help='Column to sort by (default: best_reward)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Show only top N results (default: 10, use 0 for all)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='simple',
        choices=['simple', 'grid', 'fancy_grid', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex'],
        help='Table format for display'
    )
    parser.add_argument(
        '--details',
        type=str,
        default=None,
        help='Show detailed information for a specific experiment ID'
    )
    parser.add_argument(
        '--download',
        type=str,
        default=None,
        help='Download best checkpoint from specified experiment ID'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for downloaded checkpoint or CSV export'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export comparison table to CSV'
    )
    parser.add_argument(
        '--best',
        type=int,
        default=None,
        help='Show top N best experiments by reward'
    )

    args = parser.parse_args()

    # Initialize GCS manager
    try:
        gcs_manager = GCSManager(
            bucket_name=args.bucket,
            project_id=args.project
        )
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}")
        return

    # List experiments
    if args.list:
        list_experiments(gcs_manager)
        return

    # Show experiment details
    if args.details:
        show_experiment_details(gcs_manager, args.details)
        return

    # Download checkpoint
    if args.download:
        download_best_checkpoint(gcs_manager, args.download, args.output)
        return

    # Compare experiments (default action if no specific action specified)
    if args.compare or not any([args.list, args.details, args.download, args.best]):
        experiments = gcs_manager.list_experiments()

        if not experiments:
            print("No experiments found.")
            return

        print(f"📊 Loading metrics for {len(experiments)} experiments...")
        all_metrics = load_all_metrics(gcs_manager, experiments)

        if not all_metrics:
            print("No metrics found.")
            return

        df = create_comparison_table(all_metrics, sort_by=args.sort_by)

        max_rows = args.top if args.top > 0 else None
        display_table(df, format=args.format, max_rows=max_rows)

        if args.export_csv:
            output_path = args.output or f"experiment_comparison_{args.bucket}.csv"
            save_comparison_csv(df, output_path)

        # Show summary
        print(f"\n📈 Summary:")
        print(f"   Total experiments: {len(all_metrics)}")
        print(f"   Best reward: {df['best_reward'].max():.2f}")
        print(f"   Best score (pipes): {df['best_score'].max():.0f}")
        print(f"   Average training time: {df['training_time_min'].mean():.1f} min")

    # Show best experiments
    if args.best:
        experiments = gcs_manager.list_experiments()
        all_metrics = load_all_metrics(gcs_manager, experiments)

        best_exps = find_best_experiments(all_metrics, 'best_reward', args.best)

        print(f"\n🏆 Top {args.best} Experiments by Reward:\n")
        for i, (exp_id, score, metrics) in enumerate(best_exps, 1):
            config = metrics.get('hyperparameters', {})
            print(f"{i}. {exp_id}")
            print(f"   Reward: {score:.2f}")
            print(f"   Score: {metrics.get('best_score', 0):.0f} pipes")
            print(f"   LR: {config.get('lr_start', 0):.0e} → {config.get('lr_end', 0):.0e} ({config.get('lr_schedule', '')})")
            print(f"   Hidden: {config.get('hidden_size', 0)}, Envs: {config.get('n_envs', 0)}")
            print()


if __name__ == '__main__':
    main()
