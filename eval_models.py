import os
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Import components from the repository
from chase_the_dot.env import ChaseTheDotEnv
from chase_the_dot.sac import SAC
from chase_the_dot.td3 import TD3
from chase_the_dot.ppo import PPO

def load_tb_data(logdir):
    """Loads all available tensorboard scalars from a directory."""
    events_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if 'events' in f]
    if not events_files:
        return {}
    
    event_file = events_files[0]
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    
    if 'scalars' not in ea.Tags():
        return {}
        
    tags = ea.Tags()['scalars']
    data = {}
    for scalar in tags:
        events = ea.Scalars(scalar)
        data[scalar] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    return data

def evaluate_model(algo_name, model_class, model_path, timesteps=5000):
    print(f"\n--- Evaluating {algo_name.upper()} ---")
    env = ChaseTheDotEnv(host="127.0.0.1", port=6102)
    
    try:
        env.connect()
        env.configure(speed=100, size=10.0)
    except Exception as e:
        print(f"Failed to connect to env: {e}")
        return
    
    # Initialize policy for evaluation (inference=True)
    if algo_name == "ppo":
        policy = model_class(inference=True, batch_size=256)
    else:
        policy = model_class(inference=True)
        
    policy.load(model_path)
    policy.eval()
    
    writer = SummaryWriter(log_dir=f"runs/eval/{algo_name}")
    
    obs, info = env.reset()
    batch_reward = batch_distance = batch_in_bounds = 0.0
    log_interval = 256
    
    for step_idx in tqdm(range(1, timesteps + 1), desc=f"Eval {algo_name.upper()}"):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        batch_reward += reward
        batch_distance += info["distance"]
        batch_in_bounds += int(info["in_bounds"])
        
        if step_idx % log_interval == 0:
            avg_r = batch_reward / log_interval
            avg_d = batch_distance / log_interval
            in_b_pct = (batch_in_bounds / log_interval) * 100.0
            
            writer.add_scalar("Eval/Reward", avg_r, step_idx)
            writer.add_scalar("Eval/Distance", avg_d, step_idx)
            writer.add_scalar("Eval/In_Bounds_Percent", in_b_pct, step_idx)
            
            batch_reward = batch_distance = batch_in_bounds = 0.0
            
    writer.close()
    env.close()

def plot_evaluation():
    log_dirs = {
        'SAC': 'runs/eval/sac',
        'TD3': 'runs/eval/td3',
        'PPO': 'runs/eval/ppo'
    }
    
    os.makedirs('plots', exist_ok=True)
    all_data = {algo: load_tb_data(path) for algo, path in log_dirs.items()}
    
    all_tags = set()
    for algo_data in all_data.values():
        all_tags.update(algo_data.keys())
        
    colors = {'SAC': 'tab:blue', 'TD3': 'tab:orange', 'PPO': 'tab:green'}
    
    for metric in all_tags:
        plt.figure(figsize=(8, 6))
        plotted = False
        
        for algo, data in all_data.items():
            if metric in data:
                steps = data[metric]['steps']
                values = data[metric]['values']
                
                # Smooth the data
                smoothed_values = []
                if len(values) > 0:
                    last = values[0]
                    weight = 0.6
                    for v in values:
                        smoothed_val = last * weight + (1 - weight) * v
                        smoothed_values.append(smoothed_val)
                        last = smoothed_val
                    
                    plt.plot(steps, values, color=colors[algo], alpha=0.2)
                    plt.plot(steps, smoothed_values, color=colors[algo], label=algo, linewidth=2)
                    plotted = True
                
        if plotted:
            metric_name = metric.split('/')[-1].replace('_', ' ')
            plt.title(f'Evaluation (Speed=100, Size=10): {metric_name}', fontsize=16)
            plt.xlabel('Timesteps', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            safe_name = metric.replace('/', '_').lower()
            out_path = os.path.join('plots', f'eval_{safe_name}.pdf')
            plt.tight_layout()
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            print(f"Saved {out_path}")
            
        plt.close()

def main():
    models_to_test = {
        "sac": (SAC, "models/high/sac.pt"),
        "td3": (TD3, "models/high/td3.pt"),
        "ppo": (PPO, "models/high/ppo.pt")
    }
    
    for algo_name, (model_class, model_path) in models_to_test.items():
        if os.path.exists(model_path):
            evaluate_model(algo_name, model_class, model_path, timesteps=5000)
        else:
            print(f"Warning: Model {model_path} not found. Skipping {algo_name}.")
            
    plot_evaluation()

if __name__ == '__main__':
    main()
