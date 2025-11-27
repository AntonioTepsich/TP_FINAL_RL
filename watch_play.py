import torch
import torch.nn as nn
import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import numpy as np

MODEL_PATH_NORM = "best_model_improved_full.pt"  # Usar el archivo con estadísticas
MODEL_PATH_NO_NORM = "best_model_improved.pt"  # Usar el archivo sin estadísticas


# ============================================================================
# MODELO (copiado de train_improved.py)
# ============================================================================
class VectorActorCritic(nn.Module):
    def __init__(self, obs_dim=180, n_actions=2, hidden=256):
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



def watch_agent(
    model_path=MODEL_PATH_NO_NORM,
    episodes=5,
    debug=True,
    fps=30,
    use_normalization=None,
    fast_mode=False,
    print_episode_summary=True,
    policy_mode="deterministic"  # "deterministic" | "stochastic"
):
    """
    Ejecuta el agente entrenado en el entorno Flappy Bird.

    policy_mode:
        - "deterministic": usa argmax
        - "stochastic": usa sample() (como PPO en training)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Cargar modelo ---
    print(f"📂 Cargando desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Detectar hidden size
    def get_hidden_size_from_checkpoint(ckpt):
        if 'model' in ckpt:
            w = ckpt['model'].get('shared.0.weight', None)
            if w is not None:
                return w.shape[0]
        for k in ckpt.get('model', {}):
            if k.endswith('shared.0.weight'):
                return ckpt['model'][k].shape[0]

    hidden_size = get_hidden_size_from_checkpoint(checkpoint)
    model = VectorActorCritic(obs_dim=180, n_actions=2, hidden=hidden_size)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)

    # Normalización
    has_normalization_stats = 'obs_mean' in checkpoint and 'obs_var' in checkpoint
    if use_normalization is None:
        apply_normalization = has_normalization_stats
    else:
        apply_normalization = use_normalization
        if apply_normalization and not has_normalization_stats:
            print("❌ ERROR: Se pidió normalización pero el checkpoint no tiene estadísticas.")
            return

    if apply_normalization:
        obs_mean = checkpoint['obs_mean']
        obs_var = checkpoint['obs_var']
        obs_std = np.sqrt(obs_var + 1e-8)
        print("✅ Usando normalización de observaciones")
    else:
        obs_mean = None
        obs_std = None
        print("⚠️ Ejecutando sin normalización de observaciones")

    if 'best_reward' in checkpoint:
        print(f"Best reward durante entrenamiento: {checkpoint['best_reward']:.2f}")
    print()

    # --- Crear entorno ---
    render_mode = "human" if not fast_mode else None
    env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)

    obs, info = env.reset()

    if not fast_mode:
        pygame.init()
        clock = pygame.time.Clock()

    results = []
    for ep in range(episodes):
        done = False
        ep_reward = 0
        steps = 0
        action_counts = [0, 0]  # no-flap / flap

        while not done:
            # Normalización
            if apply_normalization:
                obs_normalized = (obs - obs_mean) / obs_std
                obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # -------------------------------------------------
            # ELECCIÓN DE ACCIÓN (determinista / estocástica)
            # -------------------------------------------------
            with torch.no_grad():
                logits, value = model(obs_t)
                probs = torch.softmax(logits, dim=-1)

                if policy_mode == "stochastic":
                    # Igual que PPO en entrenamiento
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

                elif policy_mode == "deterministic" or policy_mode == "auto":
                    action = torch.argmax(probs, dim=-1).item()

                else:
                    raise ValueError(f"Modo de política inválido: {policy_mode}")

                # Debug opcional
                if debug and not fast_mode and steps % 30 == 0:
                    print(
                        f"Step {steps:3d} | Action={action} | "
                        f"P(no-flap)={probs[0][0].item():.3f}, "
                        f"P(flap)={probs[0][1].item():.3f} | "
                        f"V={value.item():+.2f}"
                    )

            # Ejecutar acción
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            if not fast_mode:
                env.render()
                clock.tick(fps)

        # Resultado episodio
        score = info.get('score', None)
        flap_pct = 100 * action_counts[1] / steps if steps > 0 else 0
        results.append({
            'reward': ep_reward,
            'steps': steps,
            'score': score,
            'flap_pct': flap_pct,
        })

        if print_episode_summary:
            print("\n" + "=" * 70)
            print(f"[EP {ep+1}] Reward={ep_reward:.2f}  Steps={steps}")
            if score is not None:
                print(f"  Caños atravesados: {score}")
            print(f"  flap%: {flap_pct:.1f}")
            print("=" * 70 + "\n")

        obs, info = env.reset()

    env.close()
    if not fast_mode:
        pygame.quit()

    return results



if __name__ == "__main__":
    # exp_20251112_205344
    # search_20251114_010504_trial008_a770b73c

    results = watch_agent(model_path="exp_old/exp_20251112_205344/checkpoints/best_model_improved_full.pt", debug=False, episodes=3, fps=30, use_normalization=True, fast_mode=True, print_episode_summary=True, policy_mode="deterministic")
    # print("Resultados:", results)