import argparse
import sys
import random
import numpy as np
import torch
from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG
from chase_the_dot.a2c import A2C
from chase_the_dot.ddpg import DDPG
from chase_the_dot.td3 import TD3
from chase_the_dot.sac import SAC
from chase_the_dot.ppo import PPO
from tqdm import tqdm

from line_profiler import profile

@profile
def main(args_list: list = None, default_algo: str = "pid") -> None:
    parser = argparse.ArgumentParser(description="Chase the Dot - Real-time Tracking Agent")
    parser.add_argument("--algo", type=str, choices=["pid", "vpg", "a2c", "ddpg", "td3", "sac", "ppo"], default=default_algo, help="Algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to run (default: 100,000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target TCP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6102, help="Target TCP port (default: 6102)")
    
    # PID parameters
    parser.add_argument("--kp", type=float, default=0.0, help="Proportional gain (PID)")
    parser.add_argument("--ki", type=float, default=0.0, help="Integral gain (PID)")
    parser.add_argument("--kd", type=float, default=0.0, help="Derivative gain (PID)")
    # RL parameters
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient (VPG)")
    parser.add_argument("--batch-size", type=int, default=64, help="Steps between updates (Default: 64)")
    
    # Env parameters
    parser.add_argument("--speed", type=int, default=None, help="Configure object speed (100-500)")
    parser.add_argument("--size", type=float, default=None, help="Configure object size (10-100)")
    parser.add_argument("--randomize-config", action="store_true", help="Randomize environment speed and size every 5000 steps")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    args = parser.parse_args(args_list)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Set global seed to {args.seed}")

    print(f"Connecting to Chase the Dot application at {args.host}:{args.port}...")
    env = ChaseTheDotEnv(host=args.host, port=args.port)
    if args.seed is not None:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        
    try:
        env.connect()
        print("Connected successfully! Auto-drain thread active.")
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    current_speed = args.speed if args.speed is not None else 300
    current_size = args.size if args.size is not None else 50.0
    print(f"Sending configuration: Speed={current_speed}, Size={current_size}%")
    env.configure(speed=current_speed, size=current_size)

    if args.algo == "pid":
        print(f"Initializing PID Policy (kp={args.kp}, ki={args.ki}, kd={args.kd})")
        policy = PID(kp=args.kp, ki=args.ki, kd=args.kd)
    elif args.algo == "a2c":
        print(f"Initializing A2C Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = A2C(lr=args.lr, gamma=args.gamma, entropy_coeff=args.entropy_coeff, batch_size=args.batch_size)
    elif args.algo == "ddpg":
        print(f"Initializing DDPG Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = DDPG(lr=args.lr, gamma=args.gamma, entropy_coeff=args.entropy_coeff, batch_size=args.batch_size)
    elif args.algo == "td3":
        print(f"Initializing TD3 Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = TD3(lr=args.lr, gamma=args.gamma, batch_size=args.batch_size)
    elif args.algo == "sac":
        print(f"Initializing SAC Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = SAC(lr=args.lr, gamma=args.gamma, batch_size=args.batch_size)
    elif args.algo == "ppo":
        print(f"Initializing PPO Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = PPO(lr=args.lr, gamma=args.gamma, entropy_coeff=args.entropy_coeff, batch_size=args.batch_size)
    else:
        print(f"Initializing VPG Policy (lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size})")
        policy = VPG(lr=args.lr, gamma=args.gamma, entropy_coeff=args.entropy_coeff, batch_size=args.batch_size)

    print(f"Starting tracking loop using {args.algo.upper()}. Press Ctrl+C to stop.")

    import os
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f"runs/{args.algo}")
    loss = 0.0
    
    try:
        step_idx = 0
        batch_reward = 0.0
        batch_distance = 0.0
        batch_dt = 0.0
        batch_in_bounds = 0
        
        state = env.receive_state(wait_for_new=True)
        
        # Initialize tqdm progress bar
        pbar = tqdm(total=args.timesteps, desc=f"Training {args.algo.upper()}", unit="step")
        
        while step_idx < args.timesteps:
            # 1. Select action
            action = policy(state)
            
            # 2. Take step in environment
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                s = env.receive_state(wait_for_new=True)
                dist = float((s[2]**2 + s[3]**2)**0.5)
                in_bounds = not (bool(s[4]) or bool(s[5]))
                reward = 1.0 - (dist / 100.) if in_bounds else -0.2 - (dist / 50.)
                obs = env.normalize(s)
                info = {"distance": dist, "in_bounds": in_bounds, "error_x": bool(s[4]), "error_y": bool(s[5]), "dt": float(s[6]), "state": s}
                terminated, truncated = False, False
            
            # Update state for the next iteration from the new observation's raw state
            state = info["state"]
            step_idx += 1
            
            # 3. Accumulate Metrics
            batch_reward += reward
            batch_distance += info.get("distance", 0.0)
            batch_dt += info.get("dt", 0.0)
            if info.get("in_bounds", False):
                batch_in_bounds += 1

            # 4. Train Policy
            step_out = policy.learn(reward)
            if isinstance(step_out, dict):
                if step_out:
                    loss = step_out.get("loss", loss)
                    for k, v in step_out.items():
                        writer.add_scalar(f"Train/{k}", v, step_idx)
            elif step_out != 0.0:
                loss = step_out

            # Randomize Config
            if args.randomize_config and step_idx > 0 and step_idx % 5000 == 0:
                current_speed = random.randint(100, 500)
                current_size = random.uniform(10.0, 100.0)
                env.configure(speed=current_speed, size=current_size)

            # 5. Logging
            if step_idx % args.batch_size == 0:
                avg_reward = batch_reward / args.batch_size
                avg_distance = batch_distance / args.batch_size
                avg_dt = batch_dt / args.batch_size
                in_bounds_percent = (batch_in_bounds / args.batch_size) * 100.0
                
                # Write standard metrics to TensorBoard
                writer.add_scalar("Metrics/Reward", avg_reward, step_idx)
                writer.add_scalar("Metrics/Loss", loss, step_idx)
                writer.add_scalar("Metrics/Distance", avg_distance, step_idx)
                writer.add_scalar("Metrics/In_Bounds_Percent", in_bounds_percent, step_idx)
                writer.add_scalar("Env/Speed", current_speed, step_idx)
                writer.add_scalar("Env/Size", current_size, step_idx)
                
                # Update tqdm metrics
                pbar.set_postfix(
                    reward=f"{avg_reward:.2f}", 
                    loss=f"{loss:.2f}",
                    dist=f"{avg_distance:.1f}",
                    in_bounds=f"{in_bounds_percent:.0f}%",
                    dt=f"{avg_dt:.3f}s"
                )
                
                # Save model periodically (e.g. every 10 * batch_size steps)
                if hasattr(policy, "save") and (step_idx % (args.batch_size * 10) == 0 or step_idx == args.timesteps):
                    os.makedirs("models", exist_ok=True)
                    model_path = os.path.join("models", f"{args.algo}_latest.pt")
                    policy.save(model_path)
                
                # Reset accumulators
                batch_reward = 0.0
                batch_distance = 0.0
                batch_dt = 0.0
                batch_in_bounds = 0
            
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        pbar.close()
        env.close()
        print("Disconnected cleanly.")

def run_pid():
    main(default_algo="pid")

def run_vpg():
    main(default_algo="vpg")

def run_a2c():
    main(default_algo="a2c")

def run_ddpg():
    main(default_algo="ddpg")

def run_td3():
    main(default_algo="td3")

def run_sac():
    main(default_algo="sac")

def run_ppo():
    main(default_algo="ppo")

if __name__ == "__main__":
    main()
