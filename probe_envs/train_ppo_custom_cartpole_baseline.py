import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ppo import PPODiagnostic  

ENV_ID = 'CartPole-v1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_steps = 2048
total_timesteps = 204_800
n_updates = total_timesteps // n_steps 
minibatch_size = 256
gamma = 0.99
lam = 0.95

N_SEEDS = 5
N_EVAL_EPISODES = 20

RESULTS_DIR = Path("runs_custom/cartpole_ppo_c")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



class MLP(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU()
        )
        self.policy = torch.nn.Linear(128, act_dim)
        self.value = torch.nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)
        return self.policy(x), self.value(x)


def make_env(seed: int):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    return env


def train_one_seed(seed: int):
    """
    Entrena PPO-C propio con una seed dada y devuelve:
    - eval_returns: vector con retornos de episodios de evaluación (long N_EVAL_EPISODES)
    """

    env = make_env(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = MLP(obs_dim, act_dim).to(device)

    ppo = PPODiagnostic(
        model,
        device=device,
        vf_coef=1.0,
        lr=5e-4,
        clip_eps=0.2,
        max_grad_norm=0.5,
        ent_coef=0.01,
        verbose=True,
    )

    episode_rewards = []
    metrics_history = {
        'reward': [],
        'explained_var': [],
        'entropy': [],
    }

    obs, _ = env.reset()

    for update in tqdm(range(n_updates), desc=f"Entrenando PPO custom (seed={seed})", unit="update"):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        total_reward_ep = 0.0
        obs_rollout_last = None
        done_rollout_last = False

        for step in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits, value = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).item()
            val = value.item()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(val)
            done_buf.append(done)

            total_reward_ep += reward

            obs_rollout_last = next_obs
            done_rollout_last = done

            obs = next_obs

            if done:
                obs, _ = env.reset()
                episode_rewards.append(total_reward_ep)
                total_reward_ep = 0.0

        if total_reward_ep > 0:
            episode_rewards.append(total_reward_ep)
            total_reward_ep = 0.0

        rew_buf = np.array(rew_buf, dtype=np.float32)
        val_buf = np.array(val_buf, dtype=np.float32)
        done_buf = np.array(done_buf, dtype=np.bool_)

        advs = np.zeros_like(rew_buf, dtype=np.float32)
        returns = np.zeros_like(rew_buf, dtype=np.float32)

        with torch.no_grad():
            if done_rollout_last:
                next_value = 0.0
            else:
                obs_t = torch.tensor(obs_rollout_last, dtype=torch.float32, device=device)
                _, v_last = model(obs_t)
                next_value = v_last.item()

        lastgaelam = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - float(done_rollout_last)
                nextvalue = next_value
            else:
                nextnonterminal = 1.0 - float(done_buf[t + 1])
                nextvalue = val_buf[t + 1]

            delta = rew_buf[t] + gamma * nextvalue * nextnonterminal - val_buf[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advs[t] = lastgaelam

        returns = advs + val_buf

        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        act_tensor = torch.tensor(np.array(act_buf), dtype=torch.int64)
        logp_tensor = torch.tensor(np.array(logp_buf), dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advs_tensor = torch.tensor(advs, dtype=torch.float32)
        vals_old_tensor = torch.tensor(val_buf, dtype=torch.float32)

        batch = [
            obs_tensor,
            act_tensor,
            logp_tensor,
            returns_tensor,
            advs_tensor,
            vals_old_tensor,
        ]

        metrics = ppo.update(batch, epochs=3, minibatch_size=minibatch_size)

        if len(episode_rewards) > 0:
            mean_last_10 = np.mean(episode_rewards[-10:])
        else:
            mean_last_10 = 0.0

        metrics_history['reward'].append(mean_last_10)
        metrics_history['explained_var'].append(metrics['explained_var'])
        metrics_history['entropy'].append(metrics['entropy'])

    env.close()

    np.savez(
        RESULTS_DIR / f'ppo_custom_results_seed{seed}.npz',
        reward=np.array(metrics_history['reward'], dtype=np.float32),
        explained_var=np.array(metrics_history['explained_var'], dtype=np.float32),
        entropy=np.array(metrics_history['entropy'], dtype=np.float32),
    )

    eval_env = gym.make(ENV_ID)
    eval_env.reset(seed=seed + 1000)  

    eval_returns = []

    model.eval()
    with torch.no_grad():
        for _ in range(N_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done = False
            ep_ret = 0.0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                logits, value = model(obs_t)

                action = torch.argmax(logits, dim=-1).item()

                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_ret += reward
                obs = next_obs

            eval_returns.append(ep_ret)

    eval_env.close()

    eval_returns = np.array(eval_returns, dtype=np.float32)
    print(f"[PPO-C][seed {seed}] mean_eval_return = {eval_returns.mean():.2f} ± {eval_returns.std():.2f}")

    return eval_returns


def main():
    all_rewards = []

    for seed in range(N_SEEDS):
        returns_seed = train_one_seed(seed)
        all_rewards.append(returns_seed)

    all_rewards = np.stack(all_rewards, axis=0) 

    np.save(RESULTS_DIR / "ppo_c_cartpole_rewards.npy", all_rewards)
    print(f" Guardado ppo_c_cartpole_rewards.npy en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
