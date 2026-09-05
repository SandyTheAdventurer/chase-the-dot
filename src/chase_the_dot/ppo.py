import numpy as np
import torch
from torch import nn
from chase_the_dot.env import normalize
from chase_the_dot.utils import mlp

class PPO(nn.Module):
    def __init__(self, actor=(64, 64, 64), critic = (64, 64, 64), sde=False, lr=0.01, gamma=0.99, entropy_coeff=0.01, clip_ratio = 0.2, ppo_epochs = 10, inference=False, batch_size = 32):
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
            self.rollout.append([feat, action, log_prob, value])

        return action.detach().numpy()

    def learn(self, reward):
        if not self.rollout:
            return 0.0
        if len(self.rollout[-1]) == 4:
            self.rollout[-1].append(reward)
        else:
            self.rollout[-1][4] += reward

        if len(self.rollout) < self.batch_size or len(self.rollout[-1]) < 5:
            return 0.0

        states, actions, old_log_probs, values, rewards = zip(*self.rollout)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs).detach()
        values = torch.stack(values).squeeze(-1).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32)

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

        self.rollout.clear()
        return total_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))