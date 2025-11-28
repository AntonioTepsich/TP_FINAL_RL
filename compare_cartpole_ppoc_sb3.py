import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas correctas según tu salida de ls
SB3_FILE = Path("runs_sb3/cartpole_ppo/sb3_ppo_cartpole_rewards.npy")
PPOC_FILE = Path("runs_custom/cartpole_ppo_c/ppo_c_cartpole_rewards.npy")


sb3 = np.load(SB3_FILE)   # [N_SEEDS, N_EVAL_EPISODES]
ppoc = np.load(PPOC_FILE) # [N_SEEDS, N_EVAL_EPISODES]

def mean_std(x):
    return x.mean(), x.std()

m_sb3, s_sb3 = mean_std(sb3)
m_ppoc, s_ppoc = mean_std(ppoc)

print(f"SB3 PPO   : {m_sb3:.1f} ± {s_sb3:.1f}")
print(f"PPO-C (tu): {m_ppoc:.1f} ± {s_ppoc:.1f}")

labels = ["PPO-C (propio)", "PPO (SB3)"]
means = [m_ppoc, m_sb3]
stds = [s_ppoc, s_sb3]

plt.bar(labels, means, yerr=stds)
plt.ylabel("Retorno medio (episodios de evaluación)")
plt.title("CartPole-v1: comparación PPO-C vs PPO (SB3)")
plt.tight_layout()
plt.savefig("cartpole_ppoc_vs_sb3.png", dpi=200)
plt.show()
