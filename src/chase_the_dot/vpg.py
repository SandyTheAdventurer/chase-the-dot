import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp

class VPG(nn.Module):
    def __init__(self, actor=(64, 64, 64), sde=False, lr=0.01, gamma=0.99, entropy_coeff=0.01, inference=False, batch_size = 32):
        super().__init__()

        self.sde = sde
        if sde:
            self.actor = mlp(7, actor, 4)
        else:
            self.actor = mlp(7, actor, 2)
            self.log_std = nn.Parameter(torch.full((2,), -2.0))
        self.gamma = gamma
        self.inference = inference
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.ptr = 0
        
        self.logprob_buf = torch.zeros(batch_size, dtype=torch.float32)
        self.entropy_buf = torch.zeros(batch_size, dtype=torch.float32)
        self.reward_buf = torch.zeros(batch_size, dtype=torch.float32)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

    def _returns(self, rewards):
        returns = torch.zeros_like(rewards)
        discounted_sum = 0
        for i in reversed(range(len(rewards))):
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            returns[i] = discounted_sum
            
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def forward(self, X):
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
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            self.logprob_buf[self.ptr] = log_prob
            self.entropy_buf[self.ptr] = entropy

        return action.detach().numpy()

    def learn(self, reward):
        if self.inference:
            return 0.0
            
        self.reward_buf[self.ptr] = reward
        self.ptr += 1

        if self.ptr < self.batch_size:
            return 0.0

        log_probs = self.logprob_buf
        entropies = self.entropy_buf
        rewards = self.reward_buf

        returns = self._returns(rewards)

        actor_loss = -(returns * log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropies.mean()

        loss = actor_loss + entropy_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.ptr = 0
        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy": entropies.mean().item()
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))