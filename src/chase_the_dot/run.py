import argparse, os, random, sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG
from chase_the_dot.a2c import A2C
from chase_the_dot.ddpg import DDPG
from chase_the_dot.td3 import TD3
from chase_the_dot.sac import SAC
from chase_the_dot.ppo import PPO

def main(args_list: list = None, default_algo: str = "pid") -> None:
    parser = argparse.ArgumentParser(description="Chase the Dot - Real-time Tracking Agent")
    parser.add_argument("--algo", type=str, choices=["pid", "vpg", "a2c", "ddpg", "td3", "sac", "ppo"], default=default_algo, help="Algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to run (default: 100,000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target TCP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6102, help="Target TCP port (default: 6102)")
    parser.add_argument("--kp", type=float, default=None, help="Proportional gain (PID, default: 0.10)")
    parser.add_argument("--ki", type=float, default=None, help="Integral gain (PID, default: 0.00)")
    parser.add_argument("--kd", type=float, default=None, help="Derivative gain (PID, default: 0.00)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient (VPG)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for off-policy updates (Default: 32)")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Rollout length for on-policy algorithms (Default: 256)")
    parser.add_argument("--speed", type=int, default=None, help="Configure object speed (100-500)")
    parser.add_argument("--size", type=float, default=None, help="Configure object size (10-100)")
    parser.add_argument("--randomize-config", action="store_true", help="Randomize environment speed and size periodically")
    parser.add_argument("--randomize-interval", type=int, default=20000, help="Timesteps between environment domain randomizations (default: 20,000)")
    parser.add_argument("--domain-expansion", "--curriculum", action="store_true", dest="domain_expansion",
                        help="Enable Domain Expansion curriculum learning (anneals from easiest domain to hardest frontier while expanding the generalization distribution)")
    parser.add_argument("--curriculum-steps", type=int, default=None,
                        help="Timesteps over which to expand the domain (default: 70%% of total timesteps)")
    parser.add_argument("--curriculum-interval", type=int, default=5000,
                        help="Timesteps between curriculum domain re-sampling (default: 5,000)")
    parser.add_argument("--curriculum-hard-ratio", type=float, default=0.5,
                        help="Probability of sampling the hardest frontier vs random within expanded domain (default: 0.5)")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation/inference mode (deterministic, no exploration noise)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint for evaluation (default: models/{algo}_latest.pt)")
    parser.add_argument("--action-scale", type=float, default=20.0, help="Residual action authority scale in pixels (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    args = parser.parse_args(args_list)

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        print(f"Set global seed to {args.seed}")

    print(f"Connecting to Chase the Dot application at {args.host}:{args.port}...")
    env = ChaseTheDotEnv(host=args.host, port=args.port, action_scale=args.action_scale)

    try:
        env.connect()
        print("Connected successfully! Auto-drain thread active.")
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr); sys.exit(1)

    curriculum_steps = args.curriculum_steps if args.curriculum_steps is not None else max(1, int(0.7 * args.timesteps))
    alpha = 0.0
    speed_min, size_min = 500, 70.0

    if args.domain_expansion and not args.eval:
        current_speed = args.speed if args.speed is not None else 500
        current_size = args.size if args.size is not None else 70.0
        print(f"Domain Expansion active: starting at easiest config (Speed={current_speed}, Size={current_size}%), expanding over {curriculum_steps} steps (hard_ratio={args.curriculum_hard_ratio})")
    else:
        current_speed = args.speed if args.speed is not None else 300
        current_size = args.size if args.size is not None else 50.0

    print(f"Sending configuration: Speed={current_speed}, Size={current_size}%")
    env.configure(speed=current_speed, size=current_size)

    shared = dict(lr=args.lr, gamma=args.gamma, inference=args.eval)
    on_policy = dict(batch_size=args.rollout_steps, entropy_coeff=args.entropy_coeff, **shared)
    off_policy = dict(batch_size=args.batch_size, **shared)
    algo_factories = {
        "pid": lambda: PID(kp=args.kp, ki=args.ki, kd=args.kd),
        "vpg": lambda: VPG(**on_policy),
        "a2c": lambda: A2C(**on_policy),
        "ppo": lambda: PPO(**on_policy),
        "ddpg": lambda: DDPG(**off_policy),
        "td3": lambda: TD3(**off_policy),
        "sac": lambda: SAC(**off_policy),
    }
    policy = algo_factories[args.algo]()
    if args.algo == "pid":
        print(f"Initializing PID Policy (kp={policy.kpx}, ki={policy.kix}, kd={policy.kdx})")
    else:
        print(f"Initializing {args.algo.upper()} Policy")
        if args.eval:
            model_path = args.model_path or os.path.join("models", f"{args.algo}_latest.pt")
            if os.path.exists(model_path):
                print(f"Loading checkpoint for evaluation: {model_path}")
                policy.load(model_path)
                policy.eval()
            else:
                print(f"Warning: Checkpoint '{model_path}' not found! Evaluating uninitialized policy.")

    action_label = "Evaluating" if args.eval else "Training"
    print(f"Starting tracking loop using {args.algo.upper()} ({action_label}). Press Ctrl+C to stop.")
    log_interval = args.rollout_steps if args.algo in ["a2c", "ppo", "vpg"] else args.batch_size
    writer = None if args.eval else SummaryWriter(log_dir=f"runs/{args.algo}")
    loss, step_idx = 0.0, 0
    batch_reward = batch_distance = batch_in_bounds = 0.0
    tot_reward = tot_distance = tot_in_bounds = 0.0

    def save_checkpoint():
        if not args.eval and hasattr(policy, "save") and step_idx > 0:
            os.makedirs("models", exist_ok=True)
            policy.save(os.path.join("models", f"{args.algo}_latest.pt"))

    pbar = None
    try:
        state = env.receive_state(wait_for_new=True)
        pbar = tqdm(total=args.timesteps, desc=f"{action_label} {args.algo.upper()}", unit="step")

        while step_idx < args.timesteps:
            action = policy(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = info["state"]
            step_idx += 1

            dist, in_b = info["distance"], int(info["in_bounds"])
            batch_reward += reward
            batch_distance += dist
            batch_in_bounds += in_b

            if args.eval:
                tot_reward += reward
                tot_distance += dist
                tot_in_bounds += in_b
            else:
                step_out = policy.learn(reward)
                if isinstance(step_out, dict):
                    loss = step_out.get("loss", loss)
                    if writer:
                        for k, v in step_out.items(): writer.add_scalar(f"Train/{k}", v, step_idx)
                elif step_out != 0.0:
                    loss = step_out

            if args.domain_expansion and not args.eval:
                alpha = min(1.0, step_idx / float(curriculum_steps))
                speed_min = int(500 - alpha * (500 - 100))
                size_min = round(70.0 - alpha * (70.0 - 10.0), 1)

                if step_idx % args.curriculum_interval == 0:
                    if random.random() < args.curriculum_hard_ratio:
                        current_speed = speed_min
                        current_size = size_min
                        mode_str = "Hard Frontier"
                    else:
                        current_speed = random.randint(speed_min, 500)
                        current_size = round(random.uniform(size_min, 70.0), 1)
                        mode_str = "Generalization"
                    env.configure(speed=current_speed, size=current_size)
                    tqdm.write(f"[{step_idx}/{args.timesteps}] Domain Expansion (alpha={alpha:.2f}, {mode_str}) -> Speed={current_speed}, Size={current_size}% | Unlocked: Speed in [{speed_min}, 500], Size in [{size_min:.1f}, 70.0]")
            elif args.randomize_config and step_idx % args.randomize_interval == 0:
                current_speed, current_size = random.randint(100, 500), random.uniform(10.0, 100.0)
                env.configure(speed=current_speed, size=current_size)

            if step_idx % log_interval == 0:
                avg_r, avg_d = batch_reward / log_interval, batch_distance / log_interval
                in_b_pct = (batch_in_bounds / log_interval) * 100.0

                if writer:
                    scalars = [
                        ("Metrics/Reward", avg_r),
                        ("Metrics/Loss", loss),
                        ("Metrics/Distance", avg_d),
                        ("Metrics/In_Bounds_Percent", in_b_pct),
                        ("Env/Speed", current_speed),
                        ("Env/Size", current_size),
                    ]
                    if args.domain_expansion and not args.eval:
                        scalars.extend([
                            ("Curriculum/Alpha", alpha),
                            ("Curriculum/Speed_Min", speed_min),
                            ("Curriculum/Size_Min", size_min),
                        ])
                    for k, v in scalars:
                        writer.add_scalar(k, v, step_idx)

                postfix = dict(dist=f"{avg_d:.1f}", in_bounds=f"{in_b_pct:.0f}%", reward=f"{avg_r:.2f}")
                if args.domain_expansion and not args.eval:
                    postfix["cfg"] = f"sp:{current_speed}|sz:{current_size:.0f}"
                if not args.eval:
                    postfix["loss"] = f"{loss:.2f}"
                pbar.set_postfix(**postfix)
                if step_idx % (log_interval * 10) == 0 or step_idx == args.timesteps:
                    save_checkpoint()
                batch_reward = batch_distance = batch_in_bounds = 0.0

            pbar.update(1)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        if pbar is not None:
            pbar.close()
        env.close()
        save_checkpoint()
        if writer is not None:
            writer.close()
        if args.eval and step_idx > 0:
            print(f"\n--- Evaluation Summary ({step_idx} steps) ---")
            print(f"In-Bounds: {tot_in_bounds / step_idx * 100:.2f}%")
            print(f"Mean Dist: {tot_distance / step_idx:.2f} px")
            print(f"Mean Rew:  {tot_reward / step_idx:.3f}")
        print("Disconnected cleanly.")

def run_pid(): main(default_algo="pid")
def run_vpg(): main(default_algo="vpg")
def run_a2c(): main(default_algo="a2c")
def run_ddpg(): main(default_algo="ddpg")
def run_td3(): main(default_algo="td3")
def run_sac(): main(default_algo="sac")
def run_ppo(): main(default_algo="ppo")

if __name__ == "__main__":
    main()
