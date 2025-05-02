import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gymnasium as gym
import gymnasium_robotics
from tensorboard.backend.event_processing import event_accumulator
import glob
import pandas as pd

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize DRND PointMaze Results')
    
    # Required arguments
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing logs to visualize')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to log_dir/visualizations)')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots interactively')
    
    return parser.parse_args()

def load_tensorboard_data(log_dir):
    """
    Load data from tensorboard logs
    
    Args:
        log_dir: Directory containing tensorboard events files
        
    Returns:
        dict: Dictionary of metrics data frames
    """
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        raise ValueError(f"No tensorboard event files found in {log_dir}")
    
    # Sort by modification time to get the latest
    event_files.sort(key=os.path.getmtime)
    latest_event_file = event_files[-1]
    
    print(f"Loading data from {latest_event_file}")
    
    # Load event file
    ea = event_accumulator.EventAccumulator(
        latest_event_file,
        size_guidance={event_accumulator.SCALARS: 0}  # Load all scalars
    )
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    
    # Group tags by category
    metrics = {}
    
    # Training metrics
    training_tags = [tag for tag in tags if tag.startswith('training/')]
    if training_tags:
        metrics['training'] = {}
        for tag in training_tags:
            events = ea.Scalars(tag)
            name = tag.split('/')[-1]
            metrics['training'][name] = pd.DataFrame(events)[['step', 'value']]
    
    # Reward metrics
    reward_tags = [tag for tag in tags if tag.startswith('rewards/')]
    if reward_tags:
        metrics['rewards'] = {}
        for tag in reward_tags:
            events = ea.Scalars(tag)
            name = tag.split('/')[-1]
            metrics['rewards'][name] = pd.DataFrame(events)[['step', 'value']]
    
    # Environment metrics (per environment)
    env_metrics = {}
    for tag in tags:
        if tag.startswith('env'):
            env_id = tag.split('/')[0]
            metric_name = tag.split('/')[-1]
            
            if env_id not in env_metrics:
                env_metrics[env_id] = {}
            
            if metric_name not in env_metrics[env_id]:
                events = ea.Scalars(tag)
                env_metrics[env_id][metric_name] = pd.DataFrame(events)[['step', 'value']]
    
    if env_metrics:
        metrics['environments'] = env_metrics
    
    return metrics

def plot_training_metrics(metrics, output_dir, show_plots=False):
    """
    Plot training metrics
    
    Args:
        metrics: Dictionary of training metrics
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    if 'training' not in metrics:
        print("No training metrics found")
        return
    
    training_metrics = metrics['training']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each metric
    for i, (name, data) in enumerate(training_metrics.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        ax.plot(data['step'], data['value'])
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Steps')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(len(training_metrics), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_reward_metrics(metrics, output_dir, show_plots=False):
    """
    Plot reward metrics
    
    Args:
        metrics: Dictionary of reward metrics
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    if 'rewards' not in metrics:
        print("No reward metrics found")
        return
    
    reward_metrics = metrics['rewards']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Reward Metrics', fontsize=16)
    
    # Plot each metric
    for name, data in reward_metrics.items():
        ax.plot(data['step'], data['value'], label=name.replace('_', ' ').title())
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reward_metrics.png'), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_environment_metrics(metrics, output_dir, show_plots=False):
    """
    Plot environment metrics
    
    Args:
        metrics: Dictionary of environment metrics
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    if 'environments' not in metrics:
        print("No environment metrics found")
        return
    
    env_metrics = metrics['environments']
    
    # Get all available metric names across environments
    all_metric_names = set()
    for env_id in env_metrics:
        all_metric_names.update(env_metrics[env_id].keys())
    
    # Plot each metric type separately
    for metric_name in all_metric_names:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'Environment Metric: {metric_name.replace("_", " ").title()}', fontsize=16)
        
        # Plot metric for each environment
        for env_id in env_metrics:
            if metric_name in env_metrics[env_id]:
                data = env_metrics[env_id][metric_name]
                ax.plot(data['step'], data['value'], label=env_id)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Only show legend if there are multiple environments
        if len(env_metrics) > 1:
            ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        filename = f'env_metric_{metric_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def create_summary_visualization(metrics, output_dir, show_plots=False):
    """
    Create a summary visualization of key metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save plots
        show_plots: Whether to show plots interactively
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DRND Training Summary', fontsize=18)
    
    # Plot rewards
    if 'rewards' in metrics:
        ax = axes[0, 0]
        ax.set_title('Rewards', fontsize=14)
        
        for name, data in metrics['rewards'].items():
            ax.plot(data['step'], data['value'], label=name.replace('_', ' ').title())
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot coverage
    if 'environments' in metrics:
        # Find coverage metric
        coverage_data = None
        for env_id, env_metrics in metrics['environments'].items():
            if 'coverage' in env_metrics:
                coverage_data = env_metrics['coverage']
                break
        
        if coverage_data is not None:
            ax = axes[0, 1]
            ax.set_title('Maze Coverage', fontsize=14)
            ax.plot(coverage_data['step'], coverage_data['value'])
            ax.set_xlabel('Steps')
            ax.set_ylabel('Coverage')
            ax.grid(True, alpha=0.3)
    
    # Plot DRND loss
    if 'training' in metrics and 'drnd_loss' in metrics['training']:
        ax = axes[1, 0]
        ax.set_title('DRND Loss', fontsize=14)
        data = metrics['training']['drnd_loss']
        ax.plot(data['step'], data['value'])
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot goal reached
    if 'environments' in metrics:
        # Find goal_reached metric
        goal_reached_data = None
        for env_id, env_metrics in metrics['environments'].items():
            if 'goal_reached' in env_metrics:
                goal_reached_data = env_metrics['goal_reached']
                break
        
        if goal_reached_data is not None:
            # Compute cumulative goals reached
            ax = axes[1, 1]
            ax.set_title('Cumulative Goals Reached', fontsize=14)
            
            # Convert to cumulative sum
            cumulative = np.cumsum(goal_reached_data['value'])
            ax.plot(goal_reached_data['step'], cumulative)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Cumulative Goals')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def visualize_heatmaps(log_dir, output_dir, show_plots=False):
    """
    Copy and organize heatmap visualizations
    
    Args:
        log_dir: Directory containing heatmaps
        output_dir: Directory to save organized heatmaps
        show_plots: Whether to show plots interactively
    """
    heatmap_dir = os.path.join(log_dir, 'heatmaps')
    if not os.path.exists(heatmap_dir):
        print("No heatmaps directory found")
        return
    
    # Find all heatmap files
    heatmap_files = glob.glob(os.path.join(heatmap_dir, 'heatmap_*.png'))
    if not heatmap_files:
        print("No heatmap files found")
        return
    
    # Sort heatmaps by episode number
    heatmap_files.sort(key=lambda x: int(x.split('_ep')[-1].split('.')[0]))
    
    # Create output directory
    heatmap_output_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_output_dir, exist_ok=True)
    
    # Create a figure showing progression of heatmaps
    num_snapshots = min(5, len(heatmap_files))
    if num_snapshots > 0:
        # Select evenly spaced heatmaps
        indices = np.linspace(0, len(heatmap_files)-1, num_snapshots, dtype=int)
        selected_heatmaps = [heatmap_files[i] for i in indices]
        
        # Create figure
        fig, axes = plt.subplots(1, num_snapshots, figsize=(4*num_snapshots, 4))
        if num_snapshots == 1:
            axes = [axes]  # Make iterable
        
        # Add each heatmap to the figure
        for i, (ax, heatmap_file) in enumerate(zip(axes, selected_heatmaps)):
            episode = int(heatmap_file.split('_ep')[-1].split('.')[0])
            img = plt.imread(heatmap_file)
            ax.imshow(img)
            ax.set_title(f'Episode {episode}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'heatmap_progression.png'), dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Copy all heatmaps to output directory
    for heatmap_file in heatmap_files:
        filename = os.path.basename(heatmap_file)
        os.system(f"cp {heatmap_file} {os.path.join(heatmap_output_dir, filename)}")
    
    print(f"Copied {len(heatmap_files)} heatmaps to {heatmap_output_dir}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.log_dir, 'visualizations')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tensorboard data
    try:
        metrics = load_tensorboard_data(args.log_dir)
        
        # Create visualizations
        plot_training_metrics(metrics, args.output_dir, args.show_plots)
        plot_reward_metrics(metrics, args.output_dir, args.show_plots)
        plot_environment_metrics(metrics, args.output_dir, args.show_plots)
        create_summary_visualization(metrics, args.output_dir, args.show_plots)
        
        print(f"Created visualizations in {args.output_dir}")
    except Exception as e:
        print(f"Error loading tensorboard data: {e}")
    
    # Visualize heatmaps
    try:
        visualize_heatmaps(args.log_dir, args.output_dir, args.show_plots)
    except Exception as e:
        print(f"Error visualizing heatmaps: {e}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()