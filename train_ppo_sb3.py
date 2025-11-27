import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time

SEED = 42

class MetricsCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.episode_rewards = []
        self.entropies = []
        self.explained_vars = []
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.start_time = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando PPO SB3", unit="step")
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])

        # Recompensas por episodio (igual que en tu custom)
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

        # Métricas de PPO que loggea SB3
        if hasattr(self.model, 'logger'):
            ent = self.model.logger.name_to_value.get('train/entropy', None)
            if ent is not None:
                self.entropies.append(ent)

            ev = self.model.logger.name_to_value.get('train/explained_variance', None)
            if ev is not None:
                self.explained_vars.append(ev)

        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"\nEntrenamiento terminado en {elapsed/60:.2f} minutos ({elapsed:.1f} segundos)")


# ======== ENV Y MODELO ========

env = gym.make('CartPole-v1')
env.reset(seed=SEED)

total_timesteps = 204_800  # 100 updates de 2048 pasos, igual que el custom

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=5e-4,
    vf_coef=1.0,
    clip_range=0.2,
    batch_size=256,
    n_steps=2048,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    policy_kwargs={'net_arch': [128, 128, 128]},
    verbose=0,
    seed=SEED,
)

callback = MetricsCallback(total_timesteps)
model.learn(total_timesteps=total_timesteps, callback=callback)

# Para comparar con tu PPO custom vamos a guardar:
# - reward: promedio móvil de 10 episodios
# - explained_var y entropy tal como las da SB3

window = 10
if len(callback.episode_rewards) >= window:
    rewards_ma = np.convolve(
        np.array(callback.episode_rewards),
        np.ones(window) / window,
        mode='valid'
    )
else:
    rewards_ma = np.array(callback.episode_rewards, dtype=np.float32)

np.savez(
    'ppo_sb3_results.npz',
    reward=rewards_ma,
    explained_var=np.array(callback.explained_vars, dtype=np.float32),
    entropy=np.array(callback.entropies, dtype=np.float32),
)
print("✅ Resultados SB3 guardados en ppo_sb3_results.npz")
