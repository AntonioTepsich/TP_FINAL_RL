import numpy as np
import matplotlib.pyplot as plt

# Cargar resultados
custom = np.load('ppo_custom_results.npz')
sb3 = np.load('ppo_sb3_results.npz')

# SB3 ya guarda reward como promedio móvil de 10 episodios
sb3_reward = sb3['reward']
custom_reward = custom['reward']

custom_explained_var = custom['explained_var']
sb3_explained_var = sb3['explained_var']

custom_entropy = custom['entropy']
sb3_entropy = sb3['entropy']

# Alinear por longitud mínima
min_len_reward = min(len(custom_reward), len(sb3_reward))
min_len_ev = min(len(custom_explained_var), len(sb3_explained_var))
min_len_ent = min(len(custom_entropy), len(sb3_entropy))

N_reward = min(50, min_len_reward)
N_ev = min(50, min_len_ev)
N_ent = min(50, min_len_ent)

# Slicing
c_reward = custom_reward[:N_reward]
s_reward = sb3_reward[:N_reward]

c_ev = custom_explained_var[:N_ev]
s_ev = sb3_explained_var[:N_ev]

c_ent = custom_entropy[:N_ent]
s_ent = sb3_entropy[:N_ent]

# ======== PLOTS ========

# Reward
plt.figure(figsize=(10, 6))
plt.plot(range(N_reward), c_reward, label='Custom PPO')
plt.plot(range(N_reward), s_reward, label='SB3 PPO')
plt.xlabel('Update')
plt.ylabel('Reward (promedio móvil / últimos 10 episodios)')
plt.title('Reward Comparison (primeros N updates)')
plt.legend()
plt.grid()
plt.savefig('reward_comparison.png')
plt.show()

# Explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(N_ev), c_ev, label='Custom PPO')
plt.plot(range(N_ev), s_ev, label='SB3 PPO')
plt.xlabel('Update')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Comparison (primeros N updates)')
plt.legend()
plt.grid()
plt.savefig('explained_var_comparison.png')
plt.show()

# Entropía
plt.figure(figsize=(10, 6))
plt.plot(range(N_ent), c_ent, label='Custom PPO')
plt.plot(range(N_ent), s_ent, label='SB3 PPO')
plt.xlabel('Update')
plt.ylabel('Entropy')
plt.title('Entropy Comparison (primeros N updates)')
plt.legend()
plt.grid()
plt.savefig('entropy_comparison.png')
plt.show()

# ======== MÉTRICAS FINALES ========

print('--- Métricas finales (últimos 10 puntos válidos) ---')
print(f"Custom PPO: Reward final promedio = {np.mean(c_reward[-10:]):.2f}")
print(f"SB3 PPO:    Reward final promedio = {np.mean(s_reward[-10:]):.2f}")

print(f"Custom PPO: Explained Var final = {c_ev[-1]:.2f}")
print(f"SB3 PPO:    Explained Var final = {s_ev[-1]:.2f}")

print(f"Custom PPO: Entropía final = {c_ent[-1]:.2f}")
print(f"SB3 PPO:    Entropía final = {s_ent[-1]:.2f}")
