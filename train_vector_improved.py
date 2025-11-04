"""
🚀 VERSIÓN MEJORADA CON TODAS LAS FIXES DE LA CHECKLIST
- Normalización de observaciones
- Métricas PPO completas
- LR y entropy schedules
- Logging detallado
- TensorBoard logging completo
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import flappy_bird_gymnasium
from ppo import PPODiagnostic
from tensorboard_logger import TensorBoardLogger
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# WRAPPERS
# ============================================================================
class NormalizeObservation(gym.ObservationWrapper):
    """Normaliza observaciones online"""
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def observation(self, obs):
        if self.running_mean is None:
            self.running_mean = np.zeros_like(obs)
            self.running_var = np.ones_like(obs)
        
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
        
        return (obs - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)

# ============================================================================
# MODELO
# ============================================================================
class VectorActorCritic(nn.Module):
    def __init__(self, obs_dim=180, n_actions=2, hidden=256):  # ← AJUSTADO A 180!
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

# ============================================================================
# VEC ENV
# ============================================================================
class VecEnv:
    def __init__(self, n_envs=16, normalize=True):
        if normalize:
            self.envs = [NormalizeObservation(gym.make("FlappyBird-v0")) for _ in range(n_envs)]
        else:
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
    
    def get_normalization_stats(self):
        """Obtiene las estadísticas de normalización del primer env (todos comparten la misma)"""
        if isinstance(self.envs[0], NormalizeObservation):
            return {
                'mean': self.envs[0].running_mean.copy(),
                'var': self.envs[0].running_var.copy(),
                'count': self.envs[0].count
            }
        return None

# ============================================================================
# GAE
# ============================================================================
def gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_adv = 0.0
    
    for t in reversed(range(T)):
        next_val = values[t+1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
    
    returns = advantages + values[:-1]
    return advantages, returns

# ============================================================================
# SCHEDULES
# ============================================================================
def linear_schedule(start, end, progress):
    """Linear interpolation from start to end"""
    return start + (end - start) * progress

def cosine_schedule(start, end, progress):
    """Cosine annealing from start to end"""
    return end + (start - end) * 0.5 * (1 + np.cos(np.pi * progress))

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("🚀 VERSIÓN MEJORADA - CON TODAS LAS FIXES + TENSORBOARD")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # CONFIGURACIÓN
    N_ENVS = 16
    T = 256
    TOTAL_STEPS = 1_000_000
    
    # Hyperparams
    LR_START = 3e-4
    LR_END = 1e-5
    ENT_START = 0.02
    ENT_END = 0.005
    
    print(f"⚙️  {N_ENVS} envs × {T} steps")
    print(f"🎯 Target: {TOTAL_STEPS:,} steps")
    print(f"📊 LR schedule: {LR_START:.0e} → {LR_END:.0e} (cosine)")
    print(f"🎲 Entropy schedule: {ENT_START:.3f} → {ENT_END:.3f} (linear)")
    print("=" * 70)
    
    # Crear TensorBoard Logger
    logger = TensorBoardLogger(comment="flappy_bird_ppo")
    
    # Log hiperparámetros iniciales como texto
    hparams_text = f"""
    ## Hyperparameters
    - **N_ENVS**: {N_ENVS}
    - **T (steps per rollout)**: {T}
    - **TOTAL_STEPS**: {TOTAL_STEPS:,}
    - **LR_START**: {LR_START:.0e}
    - **LR_END**: {LR_END:.0e}
    - **ENT_START**: {ENT_START}
    - **ENT_END**: {ENT_END}
    - **Gamma**: 0.99
    - **Lambda (GAE)**: 0.95
    - **Clip Epsilon**: 0.2
    - **VF Coef**: 0.5
    - **Max Grad Norm**: 0.5
    - **Device**: {device}
    """
    logger.log_text("Configuration", hparams_text)
    
    # Crear ambientes CON normalización
    vec = VecEnv(n_envs=N_ENVS, normalize=True)
    obs = vec.reset()
    obs_dim = obs.shape[1]
    
    print(f"\n✅ Observaciones normalizadas: dim={obs_dim}")
    
    # Modelo
    model = VectorActorCritic(obs_dim=obs_dim, n_actions=2, hidden=256)
    
    # Agent con diagnósticos (verbose=False para no spammear)
    agent = PPODiagnostic(
        model, 
        lr=LR_START, 
        clip_eps=0.2, 
        ent_coef=ENT_START,
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        device=device,
        verbose=False  # Solo mostraremos logs custom
    )
    
    global_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_scores = []  # Para trackear tubos pasados (score del juego)
    best_mean_reward = -float('inf')
    update_count = 0
    
    import time
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while global_steps < TOTAL_STEPS:
            rollout_start = time.time()
            
            # Update schedules
            progress = global_steps / TOTAL_STEPS
            current_lr = cosine_schedule(LR_START, LR_END, progress)
            current_ent = linear_schedule(ENT_START, ENT_END, progress)
            
            # Aplicar schedules
            for param_group in agent.opt.param_groups:
                param_group['lr'] = current_lr
            agent.ent_coef = current_ent
            
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
                        episode_lengths.append(info['episode']['l'])
                        # Capturar score del juego (tubos pasados)
                        if 'score' in info:
                            episode_scores.append(info['score'])

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
            
            # Update con métricas
            metrics = agent.update(batch, epochs=4, minibatch_size=1024)
            update_count += 1
            
            learn_time = time.time() - learn_start
            
            # ============================================================
            # TENSORBOARD LOGGING (cada update)
            # ============================================================
            # Métricas de PPO
            logger.log_ppo_metrics(global_steps, metrics)
            
            # Hiperparámetros
            logger.log_hyperparameters(
                global_steps, 
                current_lr, 
                current_ent,
                clip_eps=0.2,
                vf_coef=0.5
            )
            
            # Performance
            logger.log_performance(
                global_steps,
                global_steps / (time.time() - start_time),
                rollout_time,
                learn_time
            )
            
            # Advantages y Returns
            logger.log_advantages_and_returns(global_steps, advs, rets)
            
            # Estadísticas de normalización
            norm_stats = vec.get_normalization_stats()
            if norm_stats is not None:
                logger.log_normalization_stats(
                    global_steps,
                    norm_stats['mean'],
                    norm_stats['var']
                )
            
            # Métricas de episodios (si hay episodios completados)
            if episode_rewards:
                logger.log_episode_metrics(global_steps, episode_rewards, episode_lengths)
            
            # ============================================================
            # MÉTRICAS ESPECÍFICAS DE FLAPPY BIRD
            # ============================================================
            # 1. Action Distribution (detectar colapso de exploración)
            logger.log_action_distribution(global_steps, acts_t)
            
            # 2. Game Score (tubos pasados - métrica real del juego)
            if episode_scores:
                logger.log_game_score(global_steps, episode_scores)
            
            # 3. Value Predictions Quality (qué tan bien predice el crítico)
            logger.log_value_predictions(global_steps, vals_old, rets)
            
            # 4. Gradient Health (detectar explosion/vanishing)
            logger.log_gradient_health(global_steps, model)
            # ============================================================
            
            # Pesos del modelo (cada 50 updates para no saturar)
            if update_count % 50 == 0:
                logger.log_model_weights(global_steps, model)
            
            # ============================================================
            # CONSOLE LOGGING (cada 10 segundos)
            # ============================================================
            current_time = time.time()
            if current_time - last_log_time > 10.0:  # Cada 10s
                elapsed = current_time - start_time
                steps_per_sec = global_steps / elapsed
                
                if episode_rewards:
                    mean_rew = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                else:
                    mean_rew = 0.0
                
                remaining = TOTAL_STEPS - global_steps
                eta_min = (remaining / steps_per_sec) / 60 if steps_per_sec > 0 else 0
                
                print(f"\n{'='*70}")
                print(f"Steps: {global_steps:>7,}/{TOTAL_STEPS:,} ({global_steps/TOTAL_STEPS*100:>5.1f}%)")
                print(f"Episodes: {len(episode_rewards):>5} | Reward: {mean_rew:>6.2f}")
                print(f"Speed: {steps_per_sec:>6.0f} sps | ETA: {eta_min:>4.1f}m")
                print(f"Roll: {rollout_time:.2f}s | Learn: {learn_time:.2f}s")
                print(f"\n📊 Schedules:")
                print(f"   LR:      {current_lr:.2e}")
                print(f"   Entropy: {current_ent:.4f}")
                print(f"\n📈 PPO Metrics:")
                print(f"   Policy Loss:  {metrics['policy_loss']:>8.4f}")
                print(f"   Value Loss:   {metrics['value_loss']:>8.4f}")
                print(f"   Entropy:      {metrics['entropy']:>8.4f}")
                print(f"   KL Div:       {metrics['kl_div']:>8.4f}")
                print(f"   Clip Frac:    {metrics['clip_frac']:>8.4f}")
                print(f"   Ratio:        {metrics['ratio_mean']:>8.4f} ± {metrics['ratio_std']:.4f}")
                print(f"   Expl. Var:    {metrics['explained_var']:>8.4f}")
                
                # Warnings
                if abs(metrics['kl_div']) > 0.03:
                    print(f"   ⚠️  KL alto! Considera bajar LR")
                if metrics['explained_var'] < 0.2:
                    print(f"   ⚠️  EV bajo! Crítico aprende mal")
                if metrics['entropy'] < 0.01:
                    print(f"   ⚠️  Entropía muy baja! Política muy determinista")
                
                print(f"{'='*70}")
                
                last_log_time = current_time
                
                # Save best
                if episode_rewards and mean_rew > best_mean_reward:
                    best_mean_reward = mean_rew
                    
                    # Guardar solo el modelo (backward compatibility)
                    torch.save(model.state_dict(), 'best_model_improved.pt')
                    
                    # Guardar modelo + estadísticas de normalización
                    save_dict = {
                        'model': model.state_dict(),
                        'best_reward': mean_rew,
                        'global_steps': global_steps
                    }
                    
                    # Agregar estadísticas de normalización si existen
                    norm_stats = vec.get_normalization_stats()
                    if norm_stats is not None:
                        save_dict['obs_mean'] = norm_stats['mean']
                        save_dict['obs_var'] = norm_stats['var']
                        save_dict['obs_count'] = norm_stats['count']
                    
                    torch.save(save_dict, 'best_model_improved_full.pt')
                    print(f"✅ Best model saved! Reward: {mean_rew:.2f}\n")
            
            # Checkpoints
            if global_steps % 250_000 == 0 and global_steps > 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': agent.opt.state_dict(),
                    'steps': global_steps,
                    'lr': current_lr,
                    'ent_coef': current_ent,
                }, f'checkpoint_improved_{global_steps}.pt')
                print(f"💾 Checkpoint saved: {global_steps:,}\n")
    
    finally:
        vec.close()
        logger.close()
    
    # Log hiperparámetros finales para comparación
    final_metrics = {}
    if episode_rewards:
        final_metrics['final_mean_reward'] = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        final_metrics['final_max_reward'] = max(episode_rewards)
        final_metrics['total_episodes'] = len(episode_rewards)
    
    hparams = {
        'n_envs': N_ENVS,
        'rollout_steps': T,
        'lr_start': LR_START,
        'lr_end': LR_END,
        'ent_start': ENT_START,
        'ent_end': ENT_END,
        'gamma': 0.99,
        'lambda_gae': 0.95,
        'clip_eps': 0.2,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }
    
    logger.log_hparams(hparams, final_metrics)
    
    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"✅ COMPLETADO en {total_time:.2f}min")
    if episode_rewards:
        print(f"🎯 Mejor reward: {max(episode_rewards):.2f}")
        print(f"📈 Reward final (últimos 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"🚀 Velocidad promedio: {TOTAL_STEPS/(total_time*60):.0f} steps/sec")
    print(f"🔄 Updates totales: {update_count}")
    print("=" * 70)
