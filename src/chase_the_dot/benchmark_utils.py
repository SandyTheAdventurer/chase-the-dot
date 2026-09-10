"""Shared benchmark utilities for Chase the Dot evaluation scripts."""

import os
import time
from tqdm import tqdm

from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.pid import PID
from chase_the_dot.sac import SAC
from chase_the_dot.td3 import TD3
from chase_the_dot.ppo import PPO

ALGORITHMS = ["sac", "td3", "ppo", "pid"]

def load_policy(algo, action_scale=40.0, frame_stack=12):
    if algo == "pid":
        return PID(action_scale=action_scale)
    factories = {
        "sac": lambda: SAC(**shared_inference(action_scale, frame_stack)),
        "td3": lambda: TD3(**shared_inference(action_scale, frame_stack)),
        "ppo": lambda: PPO(**shared_inference(action_scale, frame_stack)),
    }
    policy = factories[algo]()
    model_path = os.path.join("models", f"{algo}_latest.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    print(f"Loading checkpoint for {algo.upper()}: {model_path}")
    policy.load(model_path)
    policy.eval()
    return policy

def shared_inference(action_scale, frame_stack):
    return dict(inference=True, frame_stack=frame_stack)

def run_benchmark(policy, timesteps, title, postfix_fn=None, on_interval=None, interval=500):
    env = ChaseTheDotEnv(action_scale=40.0, frame_stack=12)
    try:
        env.connect()
        env.configure(speed=300, size=50.0)
        obs, info = env.reset()

        tot_reward = 0.0
        tot_distance = 0.0
        tot_in_bounds = 0

        t0 = time.time()
        pbar = tqdm(total=timesteps, desc=title, unit="step")

        for step in range(timesteps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            dist = float(info["distance"])
            in_b = int(info["in_bounds"])

            tot_reward += reward
            tot_distance += dist
            tot_in_bounds += in_b

            if on_interval is not None:
                on_interval(step, tot_reward, tot_distance, tot_in_bounds, pbar)

            if (step + 1) % interval == 0 or step == timesteps - 1:
                cur_in_b = tot_in_bounds / (step + 1) * 100.0
                cur_dist = tot_distance / (step + 1)
                cur_rew = tot_reward / (step + 1)
                postfix = {"in_bounds": f"{cur_in_b:.2f}%", "dist": f"{cur_dist:.2f}px", "rew": f"{cur_rew:.3f}"}
                if postfix_fn is not None:
                    postfix.update(postfix_fn(step))
                pbar.set_postfix(**postfix)

            pbar.update(1)

        pbar.close()
        t1 = time.time()
        elapsed = t1 - t0
        throughput = timesteps / elapsed if elapsed > 0 else 0

        return {
            "timesteps": timesteps,
            "in_bounds_percent": round(tot_in_bounds / timesteps * 100.0, 2),
            "mean_distance_px": round(tot_distance / timesteps, 2),
            "mean_reward": round(tot_reward / timesteps, 3),
            "throughput_steps_sec": round(throughput, 1),
            "elapsed_seconds": round(elapsed, 1),
        }
    finally:
        env.close()
        time.sleep(2.0)
