import random
from collections import deque
import torch
from torch import nn

def mlp(in_dim, hidden_sizes, out_dim, activation=nn.ReLU):
    """Constructs a Multi-Layer Perceptron (MLP) with the given dimensions."""
    layers = []
    for s in hidden_sizes:
        layers.extend([nn.Linear(in_dim, s), activation()])
        in_dim = s
    return nn.Sequential(*layers, nn.Linear(in_dim, out_dim))

class BaseRL(nn.Module):
    """Base module providing standardized checkpoint save and load."""
    def save(self, path: str): torch.save(self.state_dict(), path)
    def load(self, path: str):
        state = torch.load(path, weights_only=True)
        cur = self.state_dict()
        adapted = {}
        for k, v in state.items():
            if k in cur and v.shape != cur[k].shape:
                if v.ndim == 2 and cur[k].ndim == 2 and v.shape[0] == cur[k].shape[0]:
                    if v.shape[1] == 9 and cur[k].shape[1] == 8:
                        v = torch.cat([v[:, :4], v[:, 5:]], dim=1)
                    elif v.shape[1] == 11 and cur[k].shape[1] == 10:
                        v = torch.cat([v[:, :4], v[:, 5:]], dim=1)
            adapted[k] = v
        self.load_state_dict(adapted)

def compute_gae(rewards, values=None, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation (GAE) or discounted returns."""
    adv = torch.zeros_like(rewards)
    gae = next_val = 0.0
    for i in reversed(range(len(rewards))):
        val = values[i] if values is not None else 0.0
        delta = rewards[i] + gamma * next_val - val
        gae = delta + gamma * gae_lambda * gae
        adv[i] = gae
        next_val = val
    returns = adv + values if values is not None else adv
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return (adv, returns) if values is not None else adv

class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    def __init__(self, maxlen=100000):
        self.buf = deque(maxlen=maxlen)
        self.cur = None

    def store_step(self, feat, action, inference=False):
        if self.cur is not None and len(self.cur) == 3:
            self.cur.append(feat)
            self.buf.append(self.cur)
        if not inference:
            self.cur = [feat, action.detach()]

    def store_reward(self, reward):
        if self.cur is not None and len(self.cur) == 2:
            self.cur.append(torch.tensor([reward], dtype=torch.float32))

    def sample(self, batch_size):
        if len(self.buf) < batch_size:
            return None
        obs, act, rew, next_obs = zip(*random.sample(self.buf, batch_size))
        return torch.stack(obs), torch.stack(act), torch.stack(rew), torch.stack(next_obs)

    def __len__(self):
        return len(self.buf)
