import numpy as np

data = np.load("runs_custom/acrobot_ppo_c/ppo_c_acrobot_rewards.npy")
mean = data.mean()
std = data.std()

print("Acrobot PPO-C → mean:", mean, " std:", std)
