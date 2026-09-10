import random
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

def compute_gae(rewards, values=None, gamma=0.99, gae_lambda=0.95, last_value=0.0):
    """Compute Generalized Advantage Estimation (GAE) or discounted returns.
    Fully vectorized — no Python loop.
    """
    T = len(rewards)
    if values is not None:
        vals = values.detach()
        next_vals = torch.empty_like(vals)
        next_vals[:-1] = vals[1:]
        next_vals[-1] = last_value
        deltas = rewards + gamma * next_vals - vals
        gam_lam_t = gamma * gae_lambda
        # Scan GAE from T-1 down to 0: gae_t = delta_t + gamma*lambda * gae_{t+1}
        gae = torch.empty_like(deltas)
        gae[-1] = deltas[-1]
        for i in range(T - 2, -1, -1):
            gae[i] = deltas[i] + gam_lam_t * gae[i + 1]
        returns = gae + vals
        if T > 1:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gae, returns
    else:
        # Discounted returns only (no values)
        disc = torch.empty(T, dtype=rewards.dtype, device=rewards.device)
        disc[-1] = rewards[-1]
        for i in range(T - 2, -1, -1):
            disc[i] = rewards[i] + gamma * disc[i + 1]
        if T > 1:
            disc = (disc - disc.mean()) / (disc.std() + 1e-8)
        return disc

class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    def __init__(self, maxlen=100000):
        self.maxlen = maxlen
        self.ptr = 0
        self.size = 0
        
        self.obs = None
        self.act = None
        self.rew = None
        self.next_obs = None
        
        self.cur_feat = None
        self.cur_act = None
        self.cur_rew = None

    def store_step(self, feat, action, inference=False):
        if self.cur_feat is not None and self.cur_rew is not None:
            # Lazy initialize buffers on the first complete transition
            if self.obs is None:
                self.obs = torch.empty((self.maxlen, *self.cur_feat.shape), dtype=torch.float32)
                self.act = torch.empty((self.maxlen, *self.cur_act.shape), dtype=torch.float32)
                self.rew = torch.empty((self.maxlen, 1), dtype=torch.float32)
                self.next_obs = torch.empty((self.maxlen, *feat.shape), dtype=torch.float32)

            self.obs[self.ptr] = self.cur_feat
            self.act[self.ptr] = self.cur_act
            self.rew[self.ptr] = self.cur_rew
            self.next_obs[self.ptr] = feat
            
            self.ptr = (self.ptr + 1) % self.maxlen
            self.size = min(self.size + 1, self.maxlen)
            
            self.cur_rew = None

        if not inference:
            self.cur_feat = feat
            self.cur_act = action.detach()

    def store_reward(self, reward):
        if self.cur_feat is not None and self.cur_rew is None:
            self.cur_rew = torch.tensor([reward], dtype=torch.float32)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        # Fast sampling using PyTorch integer indexing
        idxs = torch.randint(0, self.size, size=(batch_size,))
        return self.obs[idxs], self.act[idxs], self.rew[idxs], self.next_obs[idxs]

    def __len__(self):
        return self.size
