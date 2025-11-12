"""
Run Experiment Wrapper for PPO Flappy Bird Training
Supports single runs and hyperparameter search with GCS integration
"""

import argparse
import subprocess
import yaml
import json
import itertools
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_search_config(search_config: Dict[str, Any], output_path: str):
    """Save search configuration to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(search_config, f, default_flow_style=False)


def generate_grid_search_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations for grid search."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs


def generate_random_search_configs(
    search_space: Dict[str, List[Any]],
    n_trials: int,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Generate random configurations for random search."""
    if seed is not None:
        random.seed(seed)

    configs = []
    for _ in range(n_trials):
        config = {}
        for key, values in search_space.items():
            config[key] = random.choice(values)
        configs.append(config)

    return configs


def build_command_args(config: Dict[str, Any], base_config: Dict[str, Any]) -> List[str]:
    """Build command line arguments from configuration."""
    args = ['python', 'train_vector_improved.py']

    # Merge base config with search config (search config takes precedence)
    full_config = {**base_config, **config}

    # Convert config to command line arguments
    arg_mapping = {
        'n_envs': '--n-envs',
        'rollout_steps': '--rollout-steps',
        'total_steps': '--total-steps',
        'hidden_size': '--hidden-size',
        'lr_start': '--lr-start',
        'lr_end': '--lr-end',
        'lr_schedule': '--lr-schedule',
        'ent_start': '--ent-start',
        'ent_end': '--ent-end',
        'ent_schedule': '--ent-schedule',
        'clip_epsilon': '--clip-epsilon',
        'vf_coef': '--vf-coef',
        'max_grad_norm': '--max-grad-norm',
        'gamma': '--gamma',
        'lambda_gae': '--lambda-gae',
        'epochs_per_update': '--epochs-per-update',
        'minibatch_size': '--minibatch-size',
        'normalize_obs': '--normalize-obs',
        'gcs_bucket': '--gcs-bucket',
        'gcs_project': '--gcs-project',
        'experiment_id': '--experiment-id',
        'upload_interval': '--upload-interval',
        'log_dir': '--log-dir',
        'comment': '--comment',
        'checkpoint_interval': '--checkpoint-interval',
        'seed': '--seed',
        'device': '--device',
    }

    for key, value in full_config.items():
        if key in arg_mapping and value is not None:
            args.append(arg_mapping[key])
            args.append(str(value))

    return args


def run_single_experiment(config: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single training experiment."""
    print("\n" + "=" * 80)
    print(f"Starting experiment with config:")
    print(json.dumps(config, indent=2))
    print("=" * 80 + "\n")

    # Build command
    cmd_args = build_command_args(config, base_config)

    # Run training
    try:
        result = subprocess.run(cmd_args, check=True, capture_output=False, text=True)
        success = True
        print("\n✅ Experiment completed successfully")
    except subprocess.CalledProcessError as e:
        success = False
        print(f"\n❌ Experiment failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        raise

    return {
        'config': config,
        'success': success,
    }


def run_hyperparameter_search(
    search_config: Dict[str, Any],
    base_config: Dict[str, Any],
    output_dir: str = "search_results"
) -> List[Dict[str, Any]]:
    """Run hyperparameter search."""
    search_space = search_config.get('search_space', {})
    strategy = search_config.get('strategy', 'grid')
    n_trials = search_config.get('n_trials', None)
    seed = search_config.get('seed', None)

    # Generate configurations
    if strategy == 'grid':
        configs = generate_grid_search_configs(search_space)
        print(f"🔍 Grid Search: {len(configs)} configurations")
    elif strategy == 'random':
        if n_trials is None:
            raise ValueError("n_trials must be specified for random search")
        configs = generate_random_search_configs(search_space, n_trials, seed)
        print(f"🔍 Random Search: {len(configs)} configurations")
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = Path(output_dir) / f"search_{timestamp}"
    search_dir.mkdir(parents=True, exist_ok=True)

    # Save search configuration
    save_search_config(
        {
            'strategy': strategy,
            'search_space': search_space,
            'n_trials': n_trials if strategy == 'random' else len(configs),
            'seed': seed,
            'base_config': base_config
        },
        str(search_dir / 'search_config.yaml')
    )

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(configs)}")
        print(f"{'='*80}")

        try:
            result = run_single_experiment(config, base_config)
            results.append(result)

            # Save intermediate results
            with open(search_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

        except KeyboardInterrupt:
            print("\n⚠️ Search interrupted by user")
            break

    # Save final results summary
    with open(search_dir / 'results_summary.json', 'w') as f:
        json.dump({
            'total_experiments': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }, f, indent=2)

    print(f"\n✅ Search completed: {len(results)} experiments")
    print(f"📁 Results saved to: {search_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run PPO Flappy Bird experiments with optional hyperparameter search")

    parser.add_argument(
        '--config',
        type=str,
        default='config_template.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--search',
        action='store_true',
        help='Enable hyperparameter search mode'
    )
    parser.add_argument(
        '--search-config',
        type=str,
        default=None,
        help='Path to search configuration file (required if --search is enabled)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='search_results',
        help='Output directory for search results'
    )

    args = parser.parse_args()

    # Load base configuration
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("💡 Create a configuration file using config_template.yaml as reference")
        sys.exit(1)

    base_config = load_config(args.config)

    if args.search:
        # Hyperparameter search mode
        if args.search_config is None:
            print("❌ --search-config is required when --search is enabled")
            sys.exit(1)

        if not Path(args.search_config).exists():
            print(f"❌ Search configuration file not found: {args.search_config}")
            sys.exit(1)

        search_config = load_config(args.search_config)
        run_hyperparameter_search(search_config, base_config, args.output_dir)

    else:
        # Single run mode
        run_single_experiment({}, base_config)


if __name__ == '__main__':
    main()
