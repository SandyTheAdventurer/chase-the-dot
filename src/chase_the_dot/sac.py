import torch
from torch import nn
import numpy as np
from chase_the_dot.utils import mlp
from chase_the_dot.env import normalize
from collections import deque
import random
from line_profiler import profile

class SAC(nn.Module):
    def __init__(self, actor = (64, 64, 64), critic = (64, 64, 64), lr = 0.01, gamma = 0.99, tau = 0.005, alpha = 0.01, batch_size = 32, sde = False, inference = False):
        super().__init__()

        if not sde:
            self.actor = mlp(7, actor, 2)
            self.log_std = nn.Parameter(torch.full([2], -2.0))
        else:
            self.actor = mlp(7, actor, 4)

        self.critic1 = mlp(9, critic, 1)
        self.critic2 = mlp(9, critic, 1)
        self.target_critic1 = mlp(9, critic, 1)
        self.target_critic2 = mlp(9, critic, 1)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        
        # Automatic Entropy Tuning (alpha)
        self.target_entropy = -2.0  # Action space dimension is 2
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr, foreach=True)
        self.alpha = self.log_alpha.exp().item() # Fallback for printing/debugging

        self.inference = inference
        self.sde = sde
        self.buffer = deque(maxlen=100000)
        self.batch_size = batch_size
        self.transition = None

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, foreach=True)
        self.critic_optim = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr, foreach=True)

        # Pre-cache parameter lists for lightning-fast target updates
        self.target_critic1_params = list(self.target_critic1.parameters())
        self.critic1_params = list(self.critic1.parameters())
        self.target_critic2_params = list(self.target_critic2.parameters())
        self.critic2_params = list(self.critic2.parameters())

    @profile
    def sample(self, feat):
        if not self.sde:
            mu = self.actor(feat)
            log_sigma = self.log_std.expand_as(mu)
        else:
            out = self.actor(feat)
            mu, log_sigma = torch.split(out, 2, dim=-1)

        # Clamp log_sigma to prevent numerical instability
        log_sigma = torch.clamp(log_sigma, -20, 2)
        sigma = log_sigma.exp()

        dist = torch.distributions.Normal(mu, sigma)

        u = dist.rsample()
        action = u

        log_probs = dist.log_prob(u).sum(dim=-1, keepdim=True)

        return action, log_probs

    @profile
    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        
        with torch.no_grad():
            if self.inference:
                if not self.sde:
                    mu = self.actor(feat)
                else:
                    mu, _ = torch.split(self.actor(feat), 2, dim=-1)
                action = mu
            else:
                action, _ = self.sample(feat)

        if self.transition is not None and len(self.transition) == 3:
            self.transition.append(feat)
            self.buffer.append(self.transition)

        if not self.inference:
            self.transition = [feat, action.detach()]

        return action.detach().numpy()

    @profile
    def learn(self, reward):
        if self.transition is not None and len(self.transition) == 2:
            self.transition.append(torch.tensor([reward], dtype=torch.float32))

        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs = zip(*batch)

        obs = torch.stack(obs)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards) # [batch, 1]
        next_obs = torch.stack(next_obs)

        with torch.no_grad():
            next_action, next_log_probs = self.sample(next_obs)
            q1_next = self.target_critic1(torch.cat([next_obs, next_action], dim=1))
            q2_next = self.target_critic2(torch.cat([next_obs, next_action], dim=1))
            target_q = rewards + self.gamma * (torch.min(q1_next, q2_next) - self.alpha * next_log_probs)

        q1 = self.critic1(torch.cat([obs, actions], dim=1))
        q2 = self.critic2(torch.cat([obs, actions], dim=1))

        critic1_loss = nn.functional.mse_loss(q1, target_q)
        critic2_loss = nn.functional.mse_loss(q2, target_q)

        critic_loss = critic1_loss + critic2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor
        action, log_probs = self.sample(obs)
        q1_actor = self.critic1(torch.cat([obs, action], dim=1))
        q2_actor = self.critic2(torch.cat([obs, action], dim=1))

        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs - torch.min(q1_actor, q2_actor)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update Alpha (Temperature)
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        with torch.no_grad():
            torch._foreach_lerp_(
                self.target_critic1_params,
                self.critic1_params,
                self.tau,
            )
            torch._foreach_lerp_(
                self.target_critic2_params,
                self.critic2_params,
                self.tau,
            )

        return {
            "loss": (critic_loss.item() + actor_loss.item()) / 2.0,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))