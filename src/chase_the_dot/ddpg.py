import torch
from torch import nn
import numpy as np
from chase_the_dot.utils import mlp
from chase_the_dot.env import normalize
import random

class DDPG(nn.Module):
    def __init__(self, actor = (64, 64, 64), critic = (64, 64, 64), lr = 0.01, gamma = 0.99, tau = 0.005, entropy_coeff = 0.01, batch_size = 32, inference = False):
        super().__init__()

        self.actor = mlp(7, actor, 2)
        self.critic = mlp(9, critic, 1)

        self.target_actor = mlp(7, actor, 2)
        self.target_critic = mlp(9, critic, 1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.inference = inference
        self.buffer = []
        self.batch_size = batch_size
        self.transition = None

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = lr)

    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        action = self.actor(feat)
        
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
            target_q = rewards + self.gamma * self.target_critic(torch.cat([next_obs, next_action], dim=1))

        q = self.critic(torch.cat([obs, actions], dim=1))
        critic_loss = nn.functional.mse_loss(q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(torch.cat([obs, self.actor(obs)], dim=1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for p, target_p in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        for p, target_p in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_p.data.copy_(self.tau * p.data + (1 - self.tau) * target_p.data)

        return (critic_loss.item() + actor_loss.item()) / 2.0