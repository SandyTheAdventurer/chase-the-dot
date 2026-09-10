import torch
from torch import nn
from chase_the_dot.utils import mlp, BaseRL, ReplayBuffer
from chase_the_dot.env import normalize

class DDPG(BaseRL):
    def __init__(self, actor=(128, 128, 128), critic=(128, 128, 128), lr=0.001, gamma=0.99, tau=0.005, entropy_coeff=0.01, batch_size=32, inference=False, frame_stack=12, obs_dim=None):
        super().__init__()
        self.algo_name = "ddpg"
        self.frame_stack = int(frame_stack)
        self.obs_dim = 8 * self.frame_stack if obs_dim is None else int(obs_dim)
        self.act_dim = 2

        self.actor = mlp(self.obs_dim, actor, self.act_dim)
        self.critic = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_actor = mlp(self.obs_dim, actor, self.act_dim)
        self.target_critic = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.gamma, self.tau, self.inference, self.batch_size = gamma, tau, inference, batch_size
        self.buffer = ReplayBuffer(maxlen=100000)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, foreach=True)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr, foreach=True)

        self.target_critic_params = list(self.target_critic.parameters())
        self.critic_params = list(self.critic.parameters())
        self.target_actor_params = list(self.target_actor.parameters())
        self.actor_params = list(self.actor.parameters())

    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        action = torch.tanh(self.actor(feat))
        if not self.inference:
            action = torch.clamp(action + torch.normal(0, 0.1, size=action.shape), -1.0, 1.0)
        self.buffer.store_step(feat, action, self.inference)
        return action.detach().cpu().numpy()

    def learn(self, reward):
        if self.inference: return 0.0
        self.buffer.store_reward(reward)
        batch = self.buffer.sample(self.batch_size)
        if batch is None: return 0.0
        obs, actions, rewards, next_obs = batch

        with torch.no_grad():
            next_action = torch.tanh(self.target_actor(next_obs))
            target_q = rewards + self.gamma * self.target_critic(torch.cat([next_obs, next_action], dim=1))

        q = self.critic(torch.cat([obs, actions], dim=1))
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        actor_loss = -self.critic(torch.cat([obs, torch.tanh(self.actor(obs))], dim=1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        with torch.no_grad():
            torch._foreach_lerp_(self.target_critic_params, self.critic_params, self.tau)
            torch._foreach_lerp_(self.target_actor_params, self.actor_params, self.tau)

        return {
            "loss": (critic_loss.item() + actor_loss.item()) / 2.0,
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
