import torch
from line_profiler import profile
from torch import nn
import numpy as np
from chase_the_dot.utils import mlp, BaseRL, ReplayBuffer
from chase_the_dot.env import normalize

class SAC(BaseRL):
    def __init__(self, actor=(128, 128, 128), critic=(128, 128, 128), lr=0.001, gamma=0.99, tau=0.005, alpha=0.01, batch_size=32, sde=False, inference=False, frame_stack=12, obs_dim=None):
        super().__init__()
        self.algo_name = "sac"
        self.frame_stack = int(frame_stack)
        self.obs_dim = 8 * self.frame_stack if obs_dim is None else int(obs_dim)
        self.act_dim = 2

        self.sde = sde
        self.actor = mlp(self.obs_dim, actor, 4 if sde else self.act_dim)
        if not sde:
            self.log_std = nn.Parameter(torch.full([self.act_dim], -2.0))

        self.critic1 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.critic2 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_critic1 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_critic2 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for p in self.target_critic1.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)

        self.gamma, self.tau, self.inference, self.batch_size = gamma, tau, inference, batch_size
        self.target_entropy = -2.0
        self.log_alpha = nn.Parameter(torch.tensor([float(np.log(alpha))], dtype=torch.float32))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr, foreach=True)
        self.alpha = float(alpha)

        self.buffer = ReplayBuffer(maxlen=100000)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, foreach=True)
        self.critic_optim = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr, foreach=True)

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
            mu, log_sigma = torch.split(self.actor(feat), 2, dim=-1)

        sigma = torch.clamp(log_sigma, -20, 2).exp()
        dist = torch.distributions.Normal(mu, sigma)
        u = dist.rsample()
        action = torch.tanh(u)

        log_probs = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_probs.sum(dim=-1, keepdim=True)

    @profile
    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        with torch.no_grad():
            if self.inference:
                mu = self.actor(feat) if not self.sde else torch.split(self.actor(feat), 2, dim=-1)[0]
                action = torch.tanh(mu)
            else:
                action, _ = self.sample(feat)

        self.buffer.store_step(feat, action, self.inference)
        return action.detach().cpu().numpy()

    @profile
    def learn(self, reward):
        if self.inference: return 0.0
        self.buffer.store_reward(reward)
        batch = self.buffer.sample(self.batch_size)
        if batch is None: return 0.0
        obs, actions, rewards, next_obs = batch

        with torch.no_grad():
            next_action, next_log_probs = self.sample(next_obs)
            q1_next = self.target_critic1(torch.cat([next_obs, next_action], dim=1))
            q2_next = self.target_critic2(torch.cat([next_obs, next_action], dim=1))
            target_q = rewards + self.gamma * (torch.min(q1_next, q2_next) - self.alpha * next_log_probs)

        q1 = self.critic1(torch.cat([obs, actions], dim=1))
        q2 = self.critic2(torch.cat([obs, actions], dim=1))
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=1.0)
        self.critic_optim.step()

        # Update Actor
        action, log_probs = self.sample(obs)
        q1_actor = self.critic1(torch.cat([obs, action], dim=1))
        q2_actor = self.critic2(torch.cat([obs, action], dim=1))
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs - torch.min(q1_actor, q2_actor)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        with torch.no_grad():
            torch._foreach_lerp_(self.target_critic1_params, self.critic1_params, self.tau)
            torch._foreach_lerp_(self.target_critic2_params, self.critic2_params, self.tau)

        return {
            "loss": (critic_loss.item() + actor_loss.item()) / 2.0,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha
        }
