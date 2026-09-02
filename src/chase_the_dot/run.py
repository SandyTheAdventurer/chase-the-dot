import argparse
import sys
from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG

def main(default_algo: str = "pid") -> None:
    parser = argparse.ArgumentParser(description="Chase the Dot - Real-time Tracking Agent")
    parser.add_argument("--algo", type=str, choices=["pid", "vpg"], default=default_algo, help=f"Algorithm to use (default: {default_algo})")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to run (default: 100,000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target TCP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6102, help="Target TCP port (default: 6102)")
    
    # PID parameters
    parser.add_argument("--kp", type=float, default=0.0, help="Proportional gain (PID)")
    parser.add_argument("--ki", type=float, default=0.0, help="Integral gain (PID)")
    parser.add_argument("--kd", type=float, default=0.0, help="Derivative gain (PID)")
    # VPG parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient (VPG)")
    parser.add_argument("--batch-size", type=int, default=100, help="Steps between updates")
    
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
        print(f"Initializing Learnable PID Policy (lr={args.lr})")
        policy = PID()
    else:
        print(f"Initializing VPG Policy (lr={args.lr}, gamma={args.gamma})")
        policy = VPG(lr=args.lr, gamma=args.gamma, entropy_coeff=args.entropy_coeff)

    print(f"Starting tracking loop using {args.algo.upper()}. Press Ctrl+C to stop.")

    try:
        step_idx = 0
        in_bounds_count = 0
        rewards = []
        
        while step_idx < args.timesteps:
            # 1. Observe state
            state = env.receive_state(wait_for_new=True)
            
            # 2. Select action
            action = policy(state)
            
            # 3. Take step in environment
            # If action is None, skip sending and just get metrics
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, info = env._metrics(state)
            
            rewards.append(reward)
            step_idx += 1
            
            in_bounds = info.get("in_bounds", False)
            if in_bounds:
                in_bounds_count += 1

            # 4. Train Policy
            step_loss = policy.learn(reward)
            if step_loss != 0.0:
                loss = step_loss

            # 5. Logging
            pass

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        env.close()
        print("Disconnected cleanly.")

def run_pid():
    main(default_algo="pid")

def run_vpg():
    main(default_algo="vpg")

if __name__ == "__main__":
    main()
