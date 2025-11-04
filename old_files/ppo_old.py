import math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.out_dim = 64*7*7  # para 84x84 con convs anteriores

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.enc = ConvEncoder(in_channels)
        self.fc = nn.Sequential(nn.Linear(self.enc.out_dim, 512), nn.ReLU())
        self.pi = nn.Linear(512, n_actions)
        self.v  = nn.Linear(512, 1)

    def forward(self, x):
        z = self.fc(self.enc(x))
        return self.pi(z), self.v(z)

    def act(self, x):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v

def gae_advantages(rews, vals, dones, gamma=0.99, lam=0.95):
    T = len(rews)
    adv = torch.zeros_like(rews)
    last_adv = 0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rews[t] + gamma * vals[t+1] * mask - vals[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
    ret = adv + vals[:-1]
    return adv, ret

class PPO:
    def __init__(self, model, lr=2.5e-4, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cuda"):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.clip_eps = clip_eps; self.ent_coef = ent_coef; self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm; self.device = device

    def update(self, batch, epochs=4, minibatch_size=2048):
        obs, acts, logps_old, returns, advs, vals_old = [x.to(self.device) for x in batch]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        n = obs.size(0)
        idxs = torch.arange(n)
        for _ in range(epochs):
            perm = idxs[torch.randperm(n)]
            for i in range(0, n, minibatch_size):
                mb = perm[i:i+minibatch_size]
                logits, v = self.model(obs[mb])
                dist = Categorical(logits=logits)
                logps = dist.log_prob(acts[mb])
                ratio = torch.exp(logps - logps_old[mb])

                surr1 = ratio * advs[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = vals_old[mb] + (v.squeeze(-1) - vals_old[mb]).clamp(-self.clip_eps, self.clip_eps)
                vf_losses1 = (v.squeeze(-1) - returns[mb])**2
                vf_losses2 = (v_clipped - returns[mb])**2
                value_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef*value_loss - self.ent_coef*entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
