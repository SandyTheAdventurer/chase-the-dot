import torch
from torch import nn
import numpy as np
from chase_the_dot.utils import mlp
from chase_the_dot.env import normalize
import random

class TD3(nn.Module):
    def __init__(self, actor = (64, 64, 64), critic = (64, 64, 64), lr = 0.01, gamma = 0.99, tau = 0.005, noise_std = 0.1, noise_lmt = 0.2, policy_delay = 2, batch_size = 32, inference = False):
        super().__init__()

        self.actor = mlp(7, actor, 2)
        self.critic1 = mlp(9, critic, 1)
        self.critic2 = mlp(9, critic, 1)

        self.target_actor = mlp(7, actor, 2)
        self.target_critic1 = mlp(9, critic, 1)
        self.target_critic2 = mlp(9, critic, 1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_lmt = noise_lmt
        self.inference = inference
        self.buffer = []
        self.batch_size = batch_size
        self.transition = None
        self.steps = 0
        self.policy_delay = policy_delay
        self.last_actor_loss = 0.0

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_optim = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr = lr)

    def _smoothen(self, actions):
        noise = torch.normal(0, self.noise_std, size=actions.shape)
        noise = torch.clamp(noise, -self.noise_lmt, self.noise_lmt)
        return actions + noise

    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        action = self.actor(feat)
        
        # Add exploration noise (different from target policy smoothing noise)
        if not self.inference:
            noise = torch.normal(0, 0.1, size=action.shape)
            action = action + noise

        if self.transition is not None and len(self.transition) == 3:
            self.transition.append(feat)
            self.buffer.append(self.transition)
            if len(self.buffer) > 100000:
                self.buffer.pop(0)

        if not self.inference:
            self.transition = [feat, action.detach()]

        return action.detach().numpy()

    def learn(self, reward):
        if self.transition is not None and len(self.transition) == 2:
            self.transition.append(torch.tensor([reward], dtype=torch.float32))

        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs = zip(*batch)

        obs = torch.stack(obs)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_obs = torch.stack(next_obs)

        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            next_action = self._smoothen(next_action)
            q1_next = self.target_critic1(torch.cat([next_obs, next_action], dim=1))
            q2_next = self.target_critic2(torch.cat([next_obs, next_action], dim=1))
            next_q = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * next_q

        q1 = self.critic1(torch.cat([obs, actions], dim=1))
        q2 = self.critic2(torch.cat([obs, actions], dim=1))
        critic1_loss = nn.functional.mse_loss(q1, target_q)
        critic2_loss = nn.functional.mse_loss(q2, target_q)

        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.steps % self.policy_delay == 0:
            # TD3 typically only uses Q1 for the actor update
            q = self.critic1(torch.cat([obs, self.actor(obs)], dim=1))
            actor_loss = -q.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.last_actor_loss = actor_loss.item()

            for p, target_p in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

            for p, target_p in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

            for p, target_p in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        self.steps += 1

        return (critic_loss.item() + self.last_actor_loss) / 2.0