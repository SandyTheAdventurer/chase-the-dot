import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp

class PPO(nn.Module):
    def __init__(self, actor=(64, 64, 64), critic = (64, 64, 64), sde=False, lr=0.01, gamma=0.99, entropy_coeff=0.01, clip_ratio = 0.2, ppo_epochs = 3, inference=False, batch_size = 32):
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
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.ptr = 0
        
        self.state_buf = torch.zeros((batch_size, 7), dtype=torch.float32)
        self.action_buf = torch.zeros((batch_size, 2), dtype=torch.float32)
        self.logprob_buf = torch.zeros(batch_size, dtype=torch.float32)
        self.value_buf = torch.zeros(batch_size, dtype=torch.float32)
        self.reward_buf = torch.zeros(batch_size, dtype=torch.float32)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, foreach=True)

    def _compute_gae(self, rewards, values, gae_lambda=0.95):
        advantages = torch.zeros_like(rewards)
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * gae_lambda * gae
            advantages[i] = gae
            next_value = values[i]
            
        returns = advantages + values
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns

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
            value = self.critic(feat)
            log_prob = dist.log_prob(action).sum(dim=-1)
            self.state_buf[self.ptr] = feat
            self.action_buf[self.ptr] = action
            self.logprob_buf[self.ptr] = log_prob
            self.value_buf[self.ptr] = value.squeeze(-1)

        return action.detach().numpy()

    def learn(self, reward):
        if self.inference:
            return 0.0
            
        self.reward_buf[self.ptr] = reward
        self.ptr += 1

        if self.ptr < self.batch_size:
            return 0.0

        states = self.state_buf
        actions = self.action_buf
        old_log_probs = self.logprob_buf.detach()
        values = self.value_buf.detach()
        rewards = self.reward_buf

        advantages, returns = self._compute_gae(rewards, values)

        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            if self.sde:
                out = self.actor(states)
                mean, log_std = out[..., :2], out[..., 2:]
                std = torch.exp(torch.clamp(log_std, -20, 2))
            else:
                mean = self.actor(states)
                std = torch.exp(self.log_std)

            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropies = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            new_values = self.critic(states).squeeze(-1)
            critic_loss = nn.functional.mse_loss(new_values, returns)

            entropy_loss = -self.entropy_coeff * entropies.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss = loss.item()
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            explained_var = (1.0 - torch.var(returns - new_values) / (torch.var(returns) + 1e-8)).item()
            entropy_val = entropies.mean().item()
            actor_loss_val = actor_loss.item()
            critic_loss_val = critic_loss.item()
            
            out_metrics = {
                "loss": total_loss,
                "actor_loss": actor_loss_val,
                "critic_loss": critic_loss_val,
                "entropy": entropy_val,
                "approx_kl": approx_kl,
                "explained_var": explained_var
            }

        self.ptr = 0
        return out_metrics

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))