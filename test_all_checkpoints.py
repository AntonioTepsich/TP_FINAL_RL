import os
import glob
import torch
import numpy as np
import importlib
import re
from watch_play import watch_agent, VectorActorCritic

EXPERIMENTS_DIR = "experiments"
CHECKPOINT_NAME = "best_model_improved_full.pt"
pattern = os.path.join(EXPERIMENTS_DIR, "search_*/checkpoints/", CHECKPOINT_NAME)
checkpoints = glob.glob(pattern)

if not checkpoints:
    print("No se encontraron checkpoints para probar.")
    exit(0)


def run_and_get_results(ckpt, episodes=3):
    import watch_play
    try:
        results = watch_play.watch_agent(model_path=ckpt, debug=False, episodes=episodes, use_normalization=True, fast_mode=True, print_episode_summary=False)
        return results, None
    except Exception as e:
        return None, str(e)



results = []
EPISODES = 3
for ckpt in checkpoints:
    print(f"Probando checkpoint: {ckpt}")
    ep_results, error = run_and_get_results(ckpt, episodes=EPISODES)
    if error:
        results.append({
            'ckpt': ckpt,
            'error': error
        })
    elif ep_results and len(ep_results) > 0:
        rewards = [r['reward'] for r in ep_results]
        scores = [r['score'] for r in ep_results if r['score'] is not None]
        flaps = [r['flap_pct'] for r in ep_results]
        steps = [r['steps'] for r in ep_results]
        results.append({
            'ckpt': ckpt,
            'reward_avg': np.mean(rewards) if rewards else None,
            'score_avg': np.mean(scores) if scores else None,
            'flap_pct_avg': np.mean(flaps) if flaps else None,
            'steps_avg': np.mean(steps) if steps else None,
            'episodes': len(ep_results),
            'runs': ep_results
        })
    else:
        results.append({
            'ckpt': ckpt,
            'error': 'No se pudieron obtener resultados'
        })


# Mostrar resumen tabulado
print("\n================= RESUMEN =================\n")
print(f"{'Checkpoint':60} | {'Reward avg':>10} | {'Score avg':>10} | {'Flap %':>7} | {'Steps avg':>10} | Episodios | Error")
print("-"*130)
for r in results:
    if 'error' in r:
        print(f"{r['ckpt'][:60]:60} | {'-':>10} | {'-':>10} | {'-':>7} | {'-':>10} | {'-':>9} | {r['error']}")
    else:
        print(f"{r['ckpt'][:60]:60} | {r['reward_avg']:10.2f} | {r['score_avg']:10.2f} | {r['flap_pct_avg']:7.2f} | {r['steps_avg']:10.2f} | {r['episodes']:9d} | ")
        print("  Corridas individuales:")
        for i, run in enumerate(r['runs'], 1):
            print(f"    Run {i}: Reward={run['reward']:.2f}, Score={run['score']}, Flap%={run['flap_pct']:.2f}, Steps={run['steps']}")
