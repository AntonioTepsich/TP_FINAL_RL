import torch
import torch.nn as nn
import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import numpy as np

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

MODEL_PATH = "best_model_improved_full.pt"  # ✅ Usar el archivo con estadísticas

def watch_agent(model_path=MODEL_PATH, episodes=5, debug=True, fps=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Cargar modelo + estadísticas ---
    print(f"📂 Cargando desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = VectorActorCritic(obs_dim=180, n_actions=2)
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)
    
    # Cargar estadísticas de normalización
    if 'obs_mean' in checkpoint and 'obs_var' in checkpoint:
        obs_mean = checkpoint['obs_mean']
        obs_var = checkpoint['obs_var']
        obs_std = np.sqrt(obs_var + 1e-8)
        use_normalization = True
        print(f"✅ Modelo cargado!")
        print(f"📊 Estadísticas de normalización encontradas:")
        print(f"   Mean: min={obs_mean.min():.2f}, max={obs_mean.max():.2f}, avg={obs_mean.mean():.2f}")
        print(f"   Std:  min={obs_std.min():.2f}, max={obs_std.max():.2f}, avg={obs_std.mean():.2f}")
        if 'best_reward' in checkpoint:
            print(f"   Best reward durante entrenamiento: {checkpoint['best_reward']:.2f}")
    else:
        print("⚠️  No se encontraron estadísticas de normalización")
        print("   Usando observaciones crudas (puede no funcionar bien)")
        use_normalization = False
    
    print()

    # --- Crear el entorno ---
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    obs, info = env.reset()

    pygame.init()
    clock = pygame.time.Clock()

    for ep in range(episodes):
        done = False
        ep_reward = 0
        steps = 0
        action_counts = [0, 0]  # [no_flap, flap]
        
        while not done:
            # ✅ NORMALIZAR con las estadísticas del entrenamiento
            if use_normalization:
                obs_normalized = (obs - obs_mean) / obs_std
                obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_t)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
                
                # Debug: mostrar info cada 30 frames
                if debug and steps % 30 == 0:
                    print(f"Step {steps:3d} | Action: {action} | P(no-flap)={probs[0][0].item():.3f}, P(flap)={probs[0][1].item():.3f} | V={value.item():+.2f}")

            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            env.render()
            clock.tick(fps)

            if done:
                print(f"\n{'='*70}")
                print(f"[EP {ep+1}] Reward={ep_reward:.2f}  Steps={steps}")
                flap_pct = 100*action_counts[1]/steps if steps > 0 else 0
                print(f"  Actions: No-flap={action_counts[0]} ({100-flap_pct:.1f}%), Flap={action_counts[1]} ({flap_pct:.1f}%)")
                print(f"{'='*70}\n")
                obs, info = env.reset()
                action_counts = [0, 0]

    env.close()
    pygame.quit()


if __name__ == "__main__":
    # Opciones:
    # fps=30  -> velocidad normal
    # fps=60  -> 2x velocidad
    # fps=120 -> 4x velocidad
    # debug=True -> mostrar probabilidades cada 30 frames
    watch_agent(debug=True, episodes=3, fps=60)
