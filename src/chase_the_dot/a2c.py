import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp, BaseRL, compute_gae

class A2C(BaseRL):
    def __init__(self, actor=(128, 128, 128), critic=(128, 128, 128), sde=False, lr=0.001, gamma=0.99, entropy_coeff=0.01, inference=False, batch_size=32, frame_stack=12, obs_dim=None):
        super().__init__()
        self.algo_name = "a2c"
        self.frame_stack = int(frame_stack)
        self.obs_dim = 8 * self.frame_stack if obs_dim is None else int(obs_dim)
        self.act_dim = 2

        self.sde = sde
        self.actor = mlp(self.obs_dim, actor, 4 if sde else self.act_dim)
        if not sde:
            self.log_std = nn.Parameter(torch.full((self.act_dim,), -2.0))
        self.critic = mlp(self.obs_dim, critic, 1)
        self.gamma, self.inference, self.entropy_coeff, self.batch_size = gamma, inference, entropy_coeff, batch_size
        self._reset_buf()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

    def _reset_buf(self):
        self.ptr = 0
        self.logprobs = []
        self.entropies = []
        self.values = []
        self.reward_buf = torch.zeros(self.batch_size, dtype=torch.float32)

    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        with torch.no_grad():
            if self.sde:
                out = self.actor(feat)
                mean, log_std = out[..., :2], out[..., 2:]
                std = torch.exp(torch.clamp(log_std, -20, 2))
            else:
                mean = self.actor(feat)
                std = torch.exp(self.log_std)

            if self.inference:
                return torch.tanh(mean).cpu().numpy()

            dist = torch.distributions.Normal(mean, std)
            u = dist.sample()
            action = torch.tanh(u)

            value = self.critic(feat)
            log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            self.logprobs.append(log_prob.sum(dim=-1))
            self.entropies.append(dist.entropy().sum(dim=-1))
            self.values.append(value.squeeze(-1))
        return action.cpu().numpy()

    def learn(self, reward):
        if self.inference: return 0.0
        self.reward_buf[self.ptr] = reward
        self.ptr += 1
        if self.ptr < self.batch_size: return 0.0

        logprobs = torch.stack(self.logprobs)
        entropies = torch.stack(self.entropies)
        values = torch.stack(self.values)
        last_val = values[-1].detach().item()
        advantages, returns = compute_gae(self.reward_buf, values.detach(), gamma=self.gamma, last_value=last_val)
        actor_loss = -(advantages * logprobs).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        entropy = entropies.mean()
        loss = actor_loss + critic_loss - self.entropy_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        metrics = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }
        self._reset_buf()
        return metrics
