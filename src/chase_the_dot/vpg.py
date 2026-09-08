import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp, BaseRL, compute_gae

class VPG(BaseRL):
    def __init__(self, actor=(64, 64, 64), sde=False, lr=0.001, gamma=0.99, entropy_coeff=0.01, inference=False, batch_size=32):
        super().__init__()
        self.sde = sde
        self.actor = mlp(8, actor, 4 if sde else 2)
        if not sde:
            self.log_std = nn.Parameter(torch.full((2,), -2.0))
        self.gamma, self.inference, self.entropy_coeff, self.batch_size = gamma, inference, entropy_coeff, batch_size
        self._reset_buf()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

    def _reset_buf(self):
        self.ptr = 0
        self.logprob_buf = torch.zeros(self.batch_size, dtype=torch.float32)
        self.entropy_buf = torch.zeros(self.batch_size, dtype=torch.float32)
        self.reward_buf = torch.zeros(self.batch_size, dtype=torch.float32)

    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        if self.sde:
            out = self.actor(feat)
            mean, log_std = out[..., :2], out[..., 2:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
        else:
            mean = self.actor(feat)
            std = torch.exp(self.log_std)

        if self.inference:
            return torch.tanh(mean).detach().cpu().numpy()

        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        action = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        self.logprob_buf[self.ptr] = log_prob.sum(dim=-1)
        self.entropy_buf[self.ptr] = dist.entropy().sum(dim=-1)
        return action.detach().cpu().numpy()

    def learn(self, reward):
        if self.inference: return 0.0
        self.reward_buf[self.ptr] = reward
        self.ptr += 1
        if self.ptr < self.batch_size: return 0.0

        returns = compute_gae(self.reward_buf, gamma=self.gamma, gae_lambda=1.0)
        actor_loss = -(returns * self.logprob_buf).mean()
        entropy = self.entropy_buf.mean()
        loss = actor_loss - self.entropy_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {"loss": loss.item(), "actor_loss": actor_loss.item(), "entropy": entropy.item()}
        self._reset_buf()
        return metrics
