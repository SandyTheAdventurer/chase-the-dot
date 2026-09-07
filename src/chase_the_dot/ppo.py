import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp, BaseRL, compute_gae

class PPO(BaseRL):
    def __init__(self, actor=(64, 64, 64), critic=(64, 64, 64), sde=False, lr=0.001, gamma=0.99, entropy_coeff=0.01, clip_ratio=0.2, ppo_epochs=3, inference=False, batch_size=32):
        super().__init__()
        self.sde = sde
        self.actor = mlp(9, actor, 4 if sde else 2)
        if not sde:
            self.log_std = nn.Parameter(torch.full((2,), -2.0))
        self.critic = mlp(9, critic, 1)
        self.gamma, self.inference, self.entropy_coeff = gamma, inference, entropy_coeff
        self.clip_ratio, self.ppo_epochs, self.batch_size = clip_ratio, ppo_epochs, batch_size
        self._reset_buf()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

    def _reset_buf(self):
        self.ptr = 0
        self.state_buf = torch.zeros((self.batch_size, 9), dtype=torch.float32)
        self.action_buf = torch.zeros((self.batch_size, 2), dtype=torch.float32)
        self.logprob_buf = torch.zeros(self.batch_size, dtype=torch.float32)
        self.value_buf = torch.zeros(self.batch_size, dtype=torch.float32)
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
                action = torch.tanh(mean)
                return action.detach().cpu().numpy()

            dist = torch.distributions.Normal(mean, std)
            u = dist.sample()
            action = torch.tanh(u)

            log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
            self.state_buf[self.ptr] = feat
            self.action_buf[self.ptr] = u
            self.logprob_buf[self.ptr] = log_prob.sum(dim=-1)
            self.value_buf[self.ptr] = self.critic(feat).squeeze(-1)
        return action.detach().cpu().numpy()

    def learn(self, reward):
        if self.inference: return 0.0
        self.reward_buf[self.ptr] = reward
        self.ptr += 1
        if self.ptr < self.batch_size: return 0.0

        advantages, returns = compute_gae(self.reward_buf, self.value_buf.detach(), gamma=self.gamma)
        states, actions, old_log_probs = self.state_buf, self.action_buf, self.logprob_buf.detach()

        for _ in range(self.ppo_epochs):
            if self.sde:
                out = self.actor(states)
                mean, log_std = out[..., :2], out[..., 2:]
                std = torch.exp(torch.clamp(log_std, -20, 2))
            else:
                mean, std = self.actor(states), torch.exp(self.log_std)

            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions) - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)
            new_log_probs = new_log_probs.sum(dim=-1)
            entropies = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states).squeeze(-1)
            critic_loss = nn.functional.mse_loss(new_values, returns)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropies.mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            out_metrics = {
                "loss": loss.item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropies.mean().item(),
                "approx_kl": (old_log_probs - new_log_probs).mean().item(),
                "explained_var": (1.0 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)).item()
            }

        self._reset_buf()
        return out_metrics
