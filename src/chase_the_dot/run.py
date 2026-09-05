import argparse
import sys
import numpy as np
from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG
from chase_the_dot.a2c import A2C
from chase_the_dot.ddpg import DDPG
from chase_the_dot.td3 import TD3
from chase_the_dot.sac import SAC
from chase_the_dot.ppo import PPO
from tqdm import tqdm

def main(default_algo: str = "pid") -> None:
    parser = argparse.ArgumentParser(description="Chase the Dot - Real-time Tracking Agent")
    parser.add_argument("--algo", type=str, choices=["pid", "vpg", "a2c", "ddpg", "td3", "sac", "ppo"], default=default_algo, help=f"Algorithm to use (default: {default_algo})")
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
    parser.add_argument("--batch-size", type=int, default=256, help="Steps between updates")
    
    # Env parameters
    parser.add_argument("--speed", type=int, default=None, help="Configure object speed (100-500)")
    parser.add_argument("--size", type=float, default=None, help="Configure object size (10-100)")
    args = parser.parse_args()

    print(f"Connecting to Chase the Dot application at {args.host}:{args.port}...")
    env = ChaseTheDotEnv(host=args.host, port=args.port)
    try:
        env.connect()
        print("Connected successfully! Auto-drain thread active.")
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.speed is not None and args.size is not None:
        print(f"Sending configuration: Speed={args.speed}, Size={args.size}%")
        env.configure(speed=args.speed, size=args.size)

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

    # Initialize CSV log in logs/ directory
    import os
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"{args.algo}_log.csv")
    with open(log_filename, "w") as f:
        f.write("step,avg_reward,loss,avg_distance,avg_dt,in_bounds_percent\n")
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
                obs, reward, info = env._metrics(s)
            
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
            step_loss = policy.learn(reward)
            if step_loss != 0.0:
                loss = step_loss

            # 5. Logging
            if step_idx % args.batch_size == 0:
                avg_reward = batch_reward / args.batch_size
                avg_distance = batch_distance / args.batch_size
                avg_dt = batch_dt / args.batch_size
                in_bounds_percent = (batch_in_bounds / args.batch_size) * 100.0
                
                with open(log_filename, "a") as f:
                    f.write(f"{step_idx},{avg_reward:.4f},{loss:.4f},{avg_distance:.4f},{avg_dt:.4f},{in_bounds_percent:.1f}\n")
                
                # Update tqdm metrics
                pbar.set_postfix(
                    reward=f"{avg_reward:.2f}", 
                    loss=f"{loss:.2f}",
                    dist=f"{avg_distance:.1f}",
                    in_bounds=f"{in_bounds_percent:.0f}%"
                )
                
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
