"""
Metrics Tracking and Visualization for Training
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")

class MetricsTracker:
    """
    Tracks and stores training metrics over episodes.
    
    Stores:
    - Episode metrics (reward, deaths, infections)
    - Evaluation metrics (periodic assessments)
    - Training statistics (losses, epsilon, etc.)
    """
    
    def __init__(self):
        self.episode_metrics = defaultdict(list)
        self.evaluation_metrics = defaultdict(list)
        self.evaluation_episodes = []
        
    def add_episode(self, metrics_dict):
        """
        Add metrics from a training episode.
        
        Args:
            metrics_dict (dict): Dictionary of metrics from episode
        """
        for key, value in metrics_dict.items():
            self.episode_metrics[key].append(value)
    
    def add_evaluation(self, episode, eval_dict):
        """
        Add evaluation metrics.
        
        Args:
            episode (int): Episode number when evaluation occurred
            eval_dict (dict): Dictionary of evaluation metrics
        """
        self.evaluation_episodes.append(episode)
        for key, value in eval_dict.items():
            self.evaluation_metrics[key].append(value)
    
    def get_recent_average(self, metric_name, window=10):
        """
        Get recent moving average of a metric.
        
        Args:
            metric_name (str): Name of metric
            window (int): Window size for averaging
            
        Returns:
            float: Average value over last window episodes
        """
        if metric_name not in self.episode_metrics:
            return 0.0
        
        values = self.episode_metrics[metric_name]
        if len(values) == 0:
            return 0.0
        
        recent_values = values[-window:]
        return np.mean(recent_values)
    
    def get_metric(self, metric_name):
        """
        Get all values for a metric.
        
        Args:
            metric_name (str): Name of metric
            
        Returns:
            list: All values for this metric
        """
        return self.episode_metrics.get(metric_name, [])
    
    def save(self, filepath):
        """
        Save metrics to JSON file.
        
        Args:
            filepath (str): Path to save JSON
        """
        data = {
            'episode_metrics': {k: list(v) for k, v in self.episode_metrics.items()},
            'evaluation_metrics': {k: list(v) for k, v in self.evaluation_metrics.items()},
            'evaluation_episodes': self.evaluation_episodes
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")
    
    def load(self, filepath):
        """
        Load metrics from JSON file.
        
        Args:
            filepath (str): Path to JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_metrics = defaultdict(list, data['episode_metrics'])
        self.evaluation_metrics = defaultdict(list, data['evaluation_metrics'])
        self.evaluation_episodes = data['evaluation_episodes']
        
        print(f"Metrics loaded from: {filepath}")


def smooth_curve(values, window=10):
    """
    Smooth a curve using moving average.
    
    Args:
        values (list): Values to smooth
        window (int): Window size
        
    Returns:
        np.array: Smoothed values
    """
    if len(values) < window:
        return np.array(values)
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i+1]))
    
    return np.array(smoothed)


def plot_training_curves(metrics_tracker, save_path=None):
    """
    Create comprehensive training visualization.
    
    Args:
        metrics_tracker (MetricsTracker): Metrics to plot
        save_path (str, optional): Path to save figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Get episode numbers
    episodes = metrics_tracker.get_metric('episode')
    if len(episodes) == 0:
        print("No data to plot")
        return
    
    # 1. Episode Reward
    rewards = metrics_tracker.get_metric('episode_reward')
    if rewards:
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        axes[0, 0].plot(episodes, smooth_curve(rewards, window=20), 
                       color='darkblue', linewidth=2, label='Smoothed (20)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Reward', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Total Deaths
    deaths = metrics_tracker.get_metric('total_deaths')
    if deaths:
        axes[0, 1].plot(episodes, deaths, alpha=0.3, color='red', label='Raw')
        axes[0, 1].plot(episodes, smooth_curve(deaths, window=20),
                       color='darkred', linewidth=2, label='Smoothed (20)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Deaths')
        axes[0, 1].set_title('Deaths per Episode', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Total Infections
    infections = metrics_tracker.get_metric('total_infections')
    if infections:
        axes[0, 2].plot(episodes, infections, alpha=0.3, color='orange', label='Raw')
        axes[0, 2].plot(episodes, smooth_curve(infections, window=20),
                       color='darkorange', linewidth=2, label='Smoothed (20)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Total Infections')
        axes[0, 2].set_title('Infections per Episode', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Episode Length
    lengths = metrics_tracker.get_metric('episode_length')
    if lengths:
        axes[1, 0].plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
        axes[1, 0].plot(episodes, smooth_curve(lengths, window=20),
                       color='darkgreen', linewidth=2, label='Smoothed (20)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Days')
        axes[1, 0].set_title('Episode Length (Days)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Average Loss
    losses = metrics_tracker.get_metric('avg_loss')
    if losses:
        # Filter out zeros
        loss_episodes = [e for e, l in zip(episodes, losses) if l > 0]
        loss_values = [l for l in losses if l > 0]
        if loss_values:
            axes[1, 1].plot(loss_episodes, loss_values, alpha=0.3, color='purple', label='Raw')
            axes[1, 1].plot(loss_episodes, smooth_curve(loss_values, window=20),
                           color='darkviolet', linewidth=2, label='Smoothed (20)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Epsilon Decay
    epsilons = metrics_tracker.get_metric('epsilon')
    if epsilons:
        axes[1, 2].plot(episodes, epsilons, color='teal', linewidth=2)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].set_title('Exploration Rate (ε)', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Action Distribution
    no_lockdown = metrics_tracker.get_metric('actions_no_lockdown')
    partial = metrics_tracker.get_metric('actions_partial')
    full = metrics_tracker.get_metric('actions_full')
    if no_lockdown and partial and full:
        # Stack actions
        action_data = np.array([no_lockdown, partial, full])
        axes[2, 0].stackplot(episodes, action_data, 
                            labels=['No Lockdown', 'Partial', 'Full'],
                            colors=['green', 'orange', 'red'],
                            alpha=0.7)
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Action Count')
        axes[2, 0].set_title('Action Distribution', fontweight='bold')
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Peak Active Cases
    peak_active = metrics_tracker.get_metric('peak_active')
    if peak_active:
        axes[2, 1].plot(episodes, peak_active, alpha=0.3, color='brown', label='Raw')
        axes[2, 1].plot(episodes, smooth_curve(peak_active, window=20),
                       color='maroon', linewidth=2, label='Smoothed (20)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Peak Active Cases')
        axes[2, 1].set_title('Peak Infectious Cases', fontweight='bold')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Evaluation Results (if available)
    if metrics_tracker.evaluation_episodes:
        eval_eps = metrics_tracker.evaluation_episodes
        eval_rewards = metrics_tracker.evaluation_metrics.get('eval_avg_reward', [])
        eval_deaths = metrics_tracker.evaluation_metrics.get('eval_avg_deaths', [])
        
        if eval_rewards and eval_deaths:
            ax2 = axes[2, 2].twinx()
            
            # Plot rewards on left axis
            line1 = axes[2, 2].plot(eval_eps, eval_rewards, 'b-o', 
                                    linewidth=2, markersize=6, label='Avg Reward')
            axes[2, 2].set_xlabel('Episode')
            axes[2, 2].set_ylabel('Evaluation Reward', color='b')
            axes[2, 2].tick_params(axis='y', labelcolor='b')
            
            # Plot deaths on right axis
            line2 = ax2.plot(eval_eps, eval_deaths, 'r-s',
                            linewidth=2, markersize=6, label='Avg Deaths')
            ax2.set_ylabel('Evaluation Deaths', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[2, 2].legend(lines, labels, loc='upper right')
            
            axes[2, 2].set_title('Evaluation Performance', fontweight='bold')
            axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def create_summary_table(metrics_tracker):
    """
    Create a summary table of training statistics.
    
    Args:
        metrics_tracker (MetricsTracker): Metrics to summarize
        
    Returns:
        dict: Summary statistics
    """
    summary = {}
    
    # Episode metrics
    for metric in ['episode_reward', 'total_deaths', 'total_infections', 'peak_active', 'episode_length']:
        values = metrics_tracker.get_metric(metric)
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
            
            # Last 100 episodes
            recent = values[-100:] if len(values) >= 100 else values
            summary[f'{metric}_recent_mean'] = np.mean(recent)
    
    # Evaluation metrics
    if metrics_tracker.evaluation_episodes:
        for metric in ['eval_avg_reward', 'eval_avg_deaths', 'eval_avg_infections']:
            values = metrics_tracker.evaluation_metrics.get(metric, [])
            if values:
                summary[f'{metric}_best'] = (np.max(values) if 'reward' in metric else np.min(values))
                summary[f'{metric}_final'] = values[-1]
    
    return summary

def print_summary(metrics_tracker):
    """
    Print a human-readable summary of training.
    
    Args:
        metrics_tracker (MetricsTracker): Metrics to summarize
    """
    summary = create_summary_table(metrics_tracker)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print("\nOverall Performance:")
    if 'episode_reward_mean' in summary:
        print(f"  Average Reward: {summary['episode_reward_mean']:.2f} "
              f"(±{summary['episode_reward_std']:.2f})")
        print(f"  Recent (last 100): {summary['episode_reward_recent_mean']:.2f}")
    
    if 'total_deaths_mean' in summary:
        print(f"\n  Average Deaths: {summary['total_deaths_mean']:.1f} "
              f"(±{summary['total_deaths_std']:.1f})")
        print(f"  Recent (last 100): {summary['total_deaths_recent_mean']:.1f}")
        print(f"  Range: [{summary['total_deaths_min']:.0f}, {summary['total_deaths_max']:.0f}]")
    
    if 'total_infections_mean' in summary:
        print(f"\n  Average Infections: {summary['total_infections_mean']:.1f}")
        print(f"  Recent (last 100): {summary['total_infections_recent_mean']:.1f}")
    
    if metrics_tracker.evaluation_episodes:
        print("\nEvaluation Performance:")
        if 'eval_avg_reward_best' in summary:
            print(f"  Best Reward: {summary['eval_avg_reward_best']:.2f}")
            print(f"  Final Reward: {summary['eval_avg_reward_final']:.2f}")
        
        if 'eval_avg_deaths_best' in summary:
            print(f"  Best Deaths: {summary['eval_avg_deaths_best']:.1f}")
            print(f"  Final Deaths: {summary['eval_avg_deaths_final']:.1f}")
    
    print("="*60 + "\n")