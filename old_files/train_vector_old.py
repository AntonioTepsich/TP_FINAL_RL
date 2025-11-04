"""
VERSIÓN VECTORIAL ULTRA RÁPIDA - 50-100x MÁS RÁPIDA QUE PÍXELES
Usa observaciones simples (posición, velocidad, pipes) en vez de píxeles.
Entrena en 2-3 MINUTOS en vez de 2-3 HORAS.
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import flappy_bird_gymnasium
from ppo import PPODiagnostic as PPO

class VectorActorCritic(nn.Module):
    """Red simple para observaciones vectoriales (12 valores)"""
    def __init__(self, obs_dim=12, n_actions=2, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    
    def act(self, obs):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent, v

class VecEnv:
    """Vectoriza ambientes simples"""
    def __init__(self, n_envs=16):
        self.envs = [gym.make("FlappyBird-v0") for _ in range(n_envs)]
        self.n = n_envs
        self.episode_rewards = np.zeros(n_envs)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
    
    def reset(self):
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=i)
            obs_list.append(obs)
        return np.stack(obs_list)
    
    def step(self, actions):
        obs_list, rews, dones, truncs, infos = [], [], [], [], []
        
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            obs, r, term, trunc, info = env.step(a)
            
            self.episode_rewards[i] += r
            self.episode_lengths[i] += 1
            
            if term or trunc:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_lengths[i]
                }
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rews.append(r)
            dones.append(term)
            truncs.append(trunc)
            infos.append(info)
        
        return np.stack(obs_list), np.array(rews), np.array(dones), np.array(truncs), infos
    
    def close(self):
        for env in self.envs:
            env.close()

def gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """Calcula ventajas con GAE"""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_adv = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = values[t+1]
        else:
            next_val = values[t+1]
        
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
    
    returns = advantages + values[:-1]
    return advantages, returns

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("🚀 VERSIÓN VECTORIAL - ULTRA RÁPIDA")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # CONFIGURACIÓN
    N_ENVS = 16
    T = 256
    TOTAL_STEPS = 1_000_000
    
    print(f"⚙️  {N_ENVS} envs × {T} steps")
    print(f"💡 Observaciones: Vector simple (12 valores)")
    print(f"🎯 Target: {TOTAL_STEPS:,} steps")
    print("=" * 70)
    
    # Crear ambientes
    vec = VecEnv(n_envs=N_ENVS)
    obs = vec.reset()
    obs_dim = obs.shape[1]  # Detectar dimensión automáticamente

    # Modelo simple para vectores
    model = VectorActorCritic(obs_dim=obs_dim, n_actions=2, hidden=256)
    agent = PPO(model, lr=3e-4, clip_eps=0.2, ent_coef=0.01, 
                vf_coef=0.5, max_grad_norm=0.5, device=device)

    global_steps = 0
    episode_rewards = []
    best_mean_reward = -float('inf')
    
    import time
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while global_steps < TOTAL_STEPS:
            rollout_start = time.time()
            
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            val_buf = []
            done_buf = []
            
            # Rollout
            for _ in range(T):
                obs_t = torch.from_numpy(obs).float().to(device)
                with torch.no_grad():
                    a, logp, ent, v = model.act(obs_t)
                
                a_np = a.cpu().numpy()
                next_obs, rew, term, trunc, infos = vec.step(a_np)
                done = np.logical_or(term, trunc).astype(np.float32)
                
                for info in infos:
                    if 'episode' in info:
                        episode_rewards.append(info['episode']['r'])

                obs_buf.append(obs_t.cpu())
                act_buf.append(a.cpu())
                logp_buf.append(logp.cpu())
                rew_buf.append(torch.from_numpy(rew))
                val_buf.append(v.squeeze(-1).cpu())
                done_buf.append(torch.from_numpy(done))

                obs = next_obs
                global_steps += N_ENVS

            rollout_time = time.time() - rollout_start
            learn_start = time.time()
            
            # Calcular ventajas
            with torch.no_grad():
                last_v = model.forward(torch.from_numpy(obs).float().to(device))[1].squeeze(-1).cpu()

            obs_t = torch.stack(obs_buf).view(T*N_ENVS, obs_dim)
            acts_t = torch.stack(act_buf).view(T*N_ENVS)
            logp_t0 = torch.stack(logp_buf).view(T*N_ENVS)
            rews_t = torch.stack(rew_buf)
            vals_t = torch.stack(val_buf)
            done_t = torch.stack(done_buf)

            adv_list, ret_list, val_old_list = [], [], []
            for n in range(N_ENVS):
                adv, ret = gae_advantages(
                    rews_t[:,n], torch.cat([vals_t[:,n], last_v[n:n+1]]), done_t[:,n],
                    gamma=0.99, lam=0.95
                )
                adv_list.append(adv)
                ret_list.append(ret)
                val_old_list.append(vals_t[:,n])
            
            advs = torch.stack(adv_list).transpose(0,1).reshape(-1)
            rets = torch.stack(ret_list).transpose(0,1).reshape(-1)
            vals_old = torch.stack(val_old_list).transpose(0,1).reshape(-1)

            batch = (obs_t, acts_t, logp_t0, rets, advs, vals_old)
            metrics = agent.update(batch, epochs=4, minibatch_size=1024)
            
            learn_time = time.time() - learn_start
            
            # Logging
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - start_time
                steps_per_sec = global_steps / elapsed
                
                if episode_rewards:
                    mean_rew = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                else:
                    mean_rew = 0.0
                
                remaining = TOTAL_STEPS - global_steps
                eta_min = (remaining / steps_per_sec) / 60 if steps_per_sec > 0 else 0
                
                print("=" * 70)
                print(f"Steps: {global_steps:>7,}/{TOTAL_STEPS:,} ({global_steps/TOTAL_STEPS*100:>5.1f}%)")
                print(f"Episodes: {len(episode_rewards):>5} | Reward: {mean_rew:>6.2f}")
                print(f"Speed: {steps_per_sec:>6.0f} sps | ETA: {eta_min:>4.1f}m")
                print(f"Roll: {rollout_time:.2f}s | Learn: {learn_time:.2f}s")
                print()
                print("📈 PPO Metrics:")
                print(f"   Policy Loss: {metrics.get('policy_loss', 0):>8.4f}")
                print(f"   Value Loss:  {metrics.get('value_loss', 0):>8.4f}")
                print(f"   Entropy:     {metrics.get('entropy', 0):>8.4f}")
                print(f"   KL Div:      {metrics.get('kl_div', 0):>8.4f}")
                print(f"   Clip Frac:   {metrics.get('clip_frac', 0):>8.4f}")
                print(f"   Ratio:       {metrics.get('ratio_mean', 1):>8.4f} ± {metrics.get('ratio_std', 0):.4f}")
                print(f"   Expl. Var:   {metrics.get('explained_var', 0):>8.4f}")
                print("=" * 70)
                print()
                
                last_log_time = current_time
                
                if episode_rewards and mean_rew > best_mean_reward:
                    best_mean_reward = mean_rew
                    torch.save(model.state_dict(), 'best_model_vector.pt')
                    print(f"  ✅ Best! Reward: {mean_rew:.2f}")
            
            # Checkpoints
            if global_steps % 500_000 == 0 and global_steps > 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': agent.opt.state_dict(),
                    'steps': global_steps,
                }, f'checkpoint_vector_{global_steps}.pt')
                print(f"  💾 Checkpoint: {global_steps:,}")
    
    finally:
        vec.close()
    
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"✅ Completado en {total_time:.2f}min ({total_time*60:.0f}s)")
    if episode_rewards:
        print(f"🎯 Mejor reward: {max(episode_rewards):.2f}")
        print(f"📈 Reward final (últimos 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"🚀 Velocidad promedio: {TOTAL_STEPS/(total_time*60):.0f} steps/sec")
    print("=" * 70)
