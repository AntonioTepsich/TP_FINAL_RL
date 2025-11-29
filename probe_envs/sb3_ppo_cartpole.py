"""
Entrenamiento de PPO (Stable Baselines 3) en CartPole-v1
para usar como baseline de referencia contra tu PPO-C.

- Entrena PPO-SB3 sobre CartPole-v1 con varios seeds
- Loguea en TensorBoard
- Guarda las recompensas de evaluación en .npy
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy



TOTAL_TIMESTEPS = 200_000

N_ENVS = 4

N_SEEDS = 5

LOG_DIR = Path("runs_sb3/cartpole_ppo")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def train_and_eval_sb3(seed: int):
    """
    Entrena PPO de SB3 con un seed dado y devuelve:
    - rewards_eval: vector con retornos de episodios de evaluación
    - model_path: ruta donde se guarda el modelo entrenado
    """

    vec_env = make_vec_env(
        "CartPole-v1",
        n_envs=N_ENVS,
        seed=seed,
    )

    tb_dir = LOG_DIR / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=seed,
    )

    tb_log_name = f"sb3_ppo_cartpole_seed{seed}"

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=tb_log_name,
    )

    model_path = LOG_DIR / f"ppo_cartpole_seed{seed}.zip"
    model.save(model_path)

    eval_env = gym.make("CartPole-v1")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print(f"[SB3][seed {seed}] mean_reward = {mean_reward:.2f} ± {std_reward:.2f}")

    rewards_eval = []
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
        rewards_eval.append(ep_ret)

    eval_env.close()
    vec_env.close()

    return np.array(rewards_eval), model_path


def main():
    all_rewards = []

    for seed in range(N_SEEDS):
        rewards_seed, model_path = train_and_eval_sb3(seed)
        all_rewards.append(rewards_seed)

    all_rewards = np.stack(all_rewards, axis=0) 

    np.save(LOG_DIR / "sb3_ppo_cartpole_rewards.npy", all_rewards)

    print(f"\n[SB3] Resultados de evaluación guardados en: {LOG_DIR}")
    print("[SB3] Archivo rewards: sb3_ppo_cartpole_rewards.npy")


if __name__ == "__main__":
    main()
