import os
import glob
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from watch_play import watch_agent

def evaluate_model(path, episodes=30, policy_mode="stochastic"):
    """
    Evalúa un checkpoint en varios episodios.
    policy_mode: "stochastic" | "deterministic"
    """
    results = watch_agent(
        model_path=path,
        episodes=episodes,
        debug=False,
        fast_mode=True,
        use_normalization=None,
        print_episode_summary=False,
        policy_mode=policy_mode,
    )

    scores = [r["score"] for r in results if r["score"] is not None]

    if len(scores) == 0:
        return {
            "avg": 0.0,
            "max": 0,
            "std": 0.0,
            "weighted": 0.0,
            "raw": [],
        }

    scores = np.array(scores, dtype=float)
    avg_score = float(scores.mean())
    max_score = int(scores.max())
    std_score = float(scores.std())

    weighted = avg_score + 0.1 * max_score - 0.05 * std_score

    return {
        "avg": avg_score,
        "max": max_score,
        "std": std_score,
        "weighted": float(weighted),
        "raw": scores.tolist(),
    }


EXPERIMENTS_DIR = "exp_old"
CHECKPOINT_NAME = "best_model_improved_full.pt"
EPISODES = 10

patterns = [
    os.path.join(EXPERIMENTS_DIR, "search_*/checkpoints", CHECKPOINT_NAME),
    os.path.join(EXPERIMENTS_DIR, "exp_*/checkpoints", CHECKPOINT_NAME),
]

paths: list[str] = []
for patt in patterns:
    paths.extend(glob.glob(patt))

paths = sorted(paths)

if not paths:
    print("No se encontraron checkpoints.")
    raise SystemExit

print(f"Se encontraron {len(paths)} modelos para evaluar.")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("evaluation_results", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)


rows = []

print("\n=== EVALUANDO MODELOS ===\n")

for model_path in paths:
    print(f"Evaluando: {model_path}")

    stoch = evaluate_model(model_path, episodes=EPISODES, policy_mode="stochastic")
    det = evaluate_model(model_path, episodes=EPISODES, policy_mode="deterministic")

    rows.append(
        {
            "model_path": model_path,
            "stoch_avg": stoch["avg"],
            "stoch_max": stoch["max"],
            "stoch_std": stoch["std"],
            "stoch_weighted": stoch["weighted"],
            "det_avg": det["avg"],
            "det_max": det["max"],
            "det_std": det["std"],
            "det_weighted": det["weighted"],
            "stoch_scores": stoch["raw"],
            "det_scores": det["raw"],
        }
    )

print("\nEvaluación completa.\n")


summary_csv = os.path.join(RESULTS_DIR, "summary.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "model_path",
            "stochastic_avg",
            "stochastic_max",
            "stochastic_std",
            "stochastic_weighted",
            "deterministic_avg",
            "deterministic_max",
            "deterministic_std",
            "deterministic_weighted",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["model_path"],
                r["stoch_avg"],
                r["stoch_max"],
                r["stoch_std"],
                r["stoch_weighted"],
                r["det_avg"],
                r["det_max"],
                r["det_std"],
                r["det_weighted"],
            ]
        )

print(f"Resumen guardado en: {summary_csv}")

ranking = sorted(rows, key=lambda x: x["stoch_weighted"], reverse=True)

print("\n=== RANKING GLOBAL (modo estocástico) ===\n")
for r in ranking:
    print(
        f"{r['model_path']} → weighted={r['stoch_weighted']:.3f} | "
        f"avg={r['stoch_avg']:.2f} | max={r['stoch_max']} | std={r['stoch_std']:.2f}"
    )

labels = [
    os.path.basename(os.path.dirname(os.path.dirname(r["model_path"]))) for r in rows
]
data = [r["stoch_scores"] for r in rows]

plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
plt.boxplot(data, labels=labels, vert=True)
plt.title("Distribución de scores (modo estocástico)")
plt.ylabel("Score (pipes)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

boxplot_path = os.path.join(RESULTS_DIR, "boxplot_scores_stochastic.png")
plt.savefig(boxplot_path)
plt.close()

print(f"\n Boxplot guardado en: {boxplot_path}")

for r in rows:
    model_path = r["model_path"]
    exp_dir = os.path.dirname(os.path.dirname(model_path)) 
    exp_name = os.path.basename(exp_dir)

    out_csv = os.path.join(RESULTS_DIR, f"{exp_name}_scores.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "episode", "score"])

        for i, s in enumerate(r["stoch_scores"]):
            writer.writerow(["stochastic", i, s])

        for i, s in enumerate(r["det_scores"]):
            writer.writerow(["deterministic", i, s])

print("\n Scores individuales guardados por modelo en CSV.")
print(f" Carpeta de resultados: {RESULTS_DIR}\n")