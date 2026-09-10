import torch
from line_profiler import profile
from torch import nn
from chase_the_dot.utils import mlp, BaseRL, ReplayBuffer
from chase_the_dot.env import normalize

class TD3(BaseRL):
    def __init__(self, actor=(128, 128, 128), critic=(128, 128, 128), lr=0.001, gamma=0.99, tau=0.005, noise_std=0.2, noise_lmt=0.5, policy_delay=2, batch_size=32, inference=False, frame_stack=12, obs_dim=None):
        super().__init__()
        self.algo_name = "td3"
        self.frame_stack = int(frame_stack)
        self.obs_dim = 8 * self.frame_stack if obs_dim is None else int(obs_dim)
        self.act_dim = 2

        self.actor = mlp(self.obs_dim, actor, self.act_dim)
        self.critic1 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.critic2 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_actor = mlp(self.obs_dim, actor, self.act_dim)
        self.target_critic1 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_critic2 = mlp(self.obs_dim + self.act_dim, critic, 1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_critic1.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)

        self.gamma, self.tau, self.noise_std, self.noise_lmt = gamma, tau, noise_std, noise_lmt
        self.inference, self.batch_size, self.policy_delay = inference, batch_size, policy_delay
        self.buffer = ReplayBuffer(maxlen=100000)
        self.steps = 0
        self.last_actor_loss = 0.0

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, foreach=True)
        self.critic_optim = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr, foreach=True)

        self.target_critic1_params = list(self.target_critic1.parameters())
        self.critic1_params = list(self.critic1.parameters())
        self.target_critic2_params = list(self.target_critic2.parameters())
        self.critic2_params = list(self.critic2.parameters())
        self.target_actor_params = list(self.target_actor.parameters())
        self.actor_params = list(self.actor.parameters())

    def _smoothen(self, actions):
        noise = torch.clamp(torch.normal(0, self.noise_std, size=actions.shape), -self.noise_lmt, self.noise_lmt)
        return torch.clamp(actions + noise, -1.0, 1.0)

    @profile
    def forward(self, X):
        feat = torch.as_tensor(normalize(X), dtype=torch.float32)
        action = torch.tanh(self.actor(feat))
        if not self.inference:
            action = torch.clamp(action + torch.normal(0, 0.25, size=action.shape), -1.0, 1.0)
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
            next_action = self._smoothen(torch.tanh(self.target_actor(next_obs)))
            q1_next = self.target_critic1(torch.cat([next_obs, next_action], dim=1))
            q2_next = self.target_critic2(torch.cat([next_obs, next_action], dim=1))
            target_q = rewards + self.gamma * torch.min(q1_next, q2_next)

        q1 = self.critic1(torch.cat([obs, actions], dim=1))
        q2 = self.critic2(torch.cat([obs, actions], dim=1))
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=1.0)
        self.critic_optim.step()

        if self.steps % self.policy_delay == 0:
            q = self.critic1(torch.cat([obs, torch.tanh(self.actor(obs))], dim=1))
            actor_loss = -q.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()
            self.last_actor_loss = actor_loss.item()

            with torch.no_grad():
                torch._foreach_lerp_(self.target_critic1_params, self.critic1_params, self.tau)
                torch._foreach_lerp_(self.target_critic2_params, self.critic2_params, self.tau)
                torch._foreach_lerp_(self.target_actor_params, self.actor_params, self.tau)

        self.steps += 1
        return {
            "loss": (critic_loss.item() + self.last_actor_loss) / 2.0,
            "critic_loss": critic_loss.item(),
            "actor_loss": self.last_actor_loss
        }
