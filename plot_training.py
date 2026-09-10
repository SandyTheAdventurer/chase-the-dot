import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tb_data(logdir):
    """Loads all available tensorboard scalars from a directory."""
    events_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if 'events' in f]
    if not events_files:
        return {}
    
    # We just pick the first events file found
    event_file = events_files[0]
    
    # Load the event accumulator
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

def main():
    log_dirs = {
        'SAC': 'runs/sac',
        'TD3': 'runs/td3',
        'PPO': 'runs/ppo'
    }
    
    os.makedirs('plots', exist_ok=True)
    
    # Store all data dynamically
    all_data = {algo: load_tb_data(path) for algo, path in log_dirs.items()}
    
    # Get the union of all metrics logged across all algorithms
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
                
                # Optional: Smooth the data for cleaner plots
                smoothed_values = []
                if len(values) > 0:
                    last = values[0]
                    weight = 0.8
                    for v in values:
                        smoothed_val = last * weight + (1 - weight) * v
                        smoothed_values.append(smoothed_val)
                        last = smoothed_val
                    
                    # Plot faded raw data and bold smoothed data
                    plt.plot(steps, values, color=colors[algo], alpha=0.2)
                    plt.plot(steps, smoothed_values, color=colors[algo], label=algo, linewidth=2)
                    plotted = True
                
        if plotted:
            metric_name = metric.split('/')[-1].replace('_', ' ')
            plt.title(f'Training: {metric_name}', fontsize=16)
            plt.xlabel('Timesteps', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Save as a separate PDF
            safe_name = metric.replace('/', '_').lower()
            out_path = os.path.join('plots', f'training_{safe_name}.pdf')
            plt.tight_layout()
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            print(f"Saved {out_path}")
        
        plt.close()

if __name__ == '__main__':
    main()
