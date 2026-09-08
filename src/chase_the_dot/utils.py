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
    """Base module providing standardized checkpoint save and load with strict hyperparameter validation."""
    def save(self, path: str):
        payload = {
            "version": 2,
            "algo": getattr(self, "algo_name", self.__class__.__name__.lower()),
            "frame_stack": getattr(self, "frame_stack", 4),
            "obs_dim": getattr(self, "obs_dim", 32),
            "act_dim": getattr(self, "act_dim", 2),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            saved_algo = payload.get("algo")
            cur_algo = getattr(self, "algo_name", self.__class__.__name__.lower())
            if saved_algo is not None and cur_algo is not None and saved_algo.lower() != cur_algo.lower():
                raise ValueError(
                    f"Algorithm mismatch: Attempted to load checkpoint trained with algo='{saved_algo}' "
                    f"into model configured for algo='{cur_algo}'."
                )

            saved_fs = payload.get("frame_stack")
            cur_fs = getattr(self, "frame_stack", None)
            if saved_fs is not None and cur_fs is not None and saved_fs != cur_fs:
                raise ValueError(
                    f"Frame stack mismatch: Attempted to load checkpoint trained with frame_stack={saved_fs} "
                    f"into model configured with frame_stack={cur_fs}."
                )

            saved_obs_dim = payload.get("obs_dim")
            cur_obs_dim = getattr(self, "obs_dim", None)
            if saved_obs_dim is not None and cur_obs_dim is not None and saved_obs_dim != cur_obs_dim:
                raise ValueError(
                    f"Observation dimension mismatch: Attempted to load checkpoint trained with obs_dim={saved_obs_dim} "
                    f"into model configured with obs_dim={cur_obs_dim}."
                )

            saved_act_dim = payload.get("act_dim")
            cur_act_dim = getattr(self, "act_dim", None)
            if saved_act_dim is not None and cur_act_dim is not None and saved_act_dim != cur_act_dim:
                raise ValueError(
                    f"Action dimension mismatch: Attempted to load checkpoint trained with act_dim={saved_act_dim} "
                    f"into model configured with act_dim={cur_act_dim}."
                )

            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        cur = self.state_dict()
        for k, v in state_dict.items():
            if k in cur and v.shape != cur[k].shape:
                raise ValueError(
                    f"Tensor shape mismatch for parameter '{k}': Checkpoint tensor has shape {tuple(v.shape)}, "
                    f"but model expects shape {tuple(cur[k].shape)}. Ensure frame_stack ({getattr(self, 'frame_stack', 'unknown')}) "
                    f"and network architectures match."
                )
        self.load_state_dict(state_dict)

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
