import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize

class VPG(nn.Module):
    def __init__(self, actor=(64, 64, 64), sde=False, lr=0.01, gamma=0.99, entropy_coeff=0.01, inference=False, batch_size = 32):
        super().__init__()

        layers = []
        in_dim = 7

        for i in actor:
            layers.append(nn.Linear(in_dim, i))
            layers.append(nn.ReLU())
            in_dim = i

        self.sde = sde
        if sde:
            layers.append(nn.Linear(in_dim, 4))
        else:
            layers.append(nn.Linear(in_dim, 2))
            self.log_std = nn.Parameter(torch.full((2,), -2.0))

        self.actor = nn.Sequential(*layers)
        self.gamma = gamma
        self.inference = inference
        self.entropy_coeff = entropy_coeff
        self.rollout = []
        self.batch_size = batch_size
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def _returns(self, rewards):
        discounted_returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.gamma * g
            discounted_returns.insert(0, g)
        
        returns = torch.tensor(discounted_returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def forward(self, X):
        if X is None:
            return None
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        X_t = torch.as_tensor(X, dtype=torch.float32)

        if X_t[0] > 2.0:
            gx, gy = float(X_t[0]), float(X_t[1])
        else:
            gx, gy = float(X_t[0] * 1000.), float(X_t[1] * 1000.)
            
        if self.sde:
            out = self.actor(feat)
            mean, log_std = out[..., :2], out[..., 2:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
        else:
            mean = self.actor(feat)
            std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        if not self.inference:
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            self.rollout.append([log_prob, entropy])

        target_x = max(-50, min(950, int(round(gx + float(action[0].item()) * 100.0))))
        target_y = max(-50, min(750, int(round(gy + float(action[1].item()) * 100.0))))
        return np.array([target_x, target_y])

    def learn(self, reward):
        if not self.rollout:
            return 0.0
        if len(self.rollout[-1]) == 2:
            self.rollout[-1].append(reward)
        else:
            self.rollout[-1][2] += reward

        if len(self.rollout) < self.batch_size or len(self.rollout[-1]) < 3:
            return 0.0
        log_probs, entropies, rewards = zip(*self.rollout)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        returns = self._returns(rewards)

        actor_loss = -(returns * log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropies.mean()

        loss = actor_loss + entropy_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.rollout.clear()
        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))