import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp

class A2C(nn.Module):
    def __init__(self, actor=(64, 64, 64), critic = (64, 64, 64), sde=False, lr=0.01, gamma=0.99, entropy_coeff=0.01, inference=False, batch_size = 32):
        super().__init__()

        self.sde = sde
        if sde:
            self.actor = mlp(7, actor, 4)
        else:
            self.actor = mlp(7, actor, 2)
            self.log_std = nn.Parameter(torch.full((2,), -2.0))
            
        self.critic = mlp(7, critic, 1)
        self.gamma = gamma
        self.inference = inference
        self.entropy_coeff = entropy_coeff
        self.rollout = []
        self.batch_size = batch_size
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def _compute_gae(self, rewards, values, gae_lambda=0.95):
        advantages = []
        gae = 0
        next_value = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * gae_lambda * gae
            advantages.append(gae)
            next_value = v
            
        advantages.reverse()
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        returns = advantages + values
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns

    def forward(self, X):
        if X is None:
            return None
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)

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
            value = self.critic(feat)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            self.rollout.append([log_prob, entropy, value])

        return action.detach().numpy()

    def learn(self, reward):
        if not self.rollout:
            return 0.0
        if len(self.rollout[-1]) == 3:
            self.rollout[-1].append(reward)
        else:
            self.rollout[-1][3] += reward

        if len(self.rollout) < self.batch_size or len(self.rollout[-1]) < 4:
            return 0.0
        log_probs, entropies, values, rewards = zip(*self.rollout)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values).squeeze(-1)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        advantages, returns = self._compute_gae(rewards, values.detach())

        actor_loss = -(advantages * log_probs).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -self.entropy_coeff * entropies.mean()

        loss = actor_loss + critic_loss + entropy_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.rollout.clear()
        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))