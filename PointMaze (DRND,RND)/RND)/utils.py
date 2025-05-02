import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
from datetime import datetime
import json

class ExplorationMetrics:
    """
    Track and visualize exploration metrics for maze environments
    """
    def __init__(self, grid_size=30, maze_bounds=None, log_dir=None, heatmap_interval=50, maze_map=None):
        """
        Initialize exploration metrics tracking.
        
        Args:
            grid_size (int): Resolution of the grid for tracking visitation
            maze_bounds (dict): Dictionary with 'min_x', 'max_x', 'min_y', 'max_y' 
                               or None to determine dynamically
            log_dir (str): Directory to save visualizations
            heatmap_interval (int): Generate heatmap every n episodes
            maze_map (list): 2D list representing the maze layout (1=wall, 0=free space)
        """
        self.grid_size = grid_size
        self.maze_bounds = maze_bounds or {'min_x': -4, 'max_x': 4, 'min_y': -4, 'max_y': 4}
        self.dynamic_bounds = maze_bounds is None
        self.heatmap_interval = heatmap_interval
        self.maze_map = maze_map
        
        # Initialize metrics
        self.visited_cells = np.zeros((grid_size, grid_size))
        self.visit_counts = np.zeros((grid_size, grid_size))
        self.max_distance = 0.0
        self.min_goal_distance = float('inf')
        self.goal_proximity_count = 0
        self.starting_positions = []
        self.log_dir = log_dir
        self.goal_position = None

        # Initialize per-episode metrics
        self.episode_positions = []
        self.episode_max_distance = 0.0
        self.episode_min_goal_distance = float('inf')
        self.episode_goal_reached = False
        self.episode_count = 0
        
        # Create directory for heatmaps if needed
        if log_dir:
            self.heatmap_dir = os.path.join(log_dir, 'heatmaps')
            os.makedirs(self.heatmap_dir, exist_ok=True)
    
    def reset(self):
        """Reset per-episode metrics."""
        self.episode_positions = []  # Clear the positions list
        self.episode_max_distance = 0.0
        self.episode_min_goal_distance = float('inf')
        self.episode_goal_reached = False
    
    def update_bounds(self, position):
        """
        Dynamically update maze bounds based on observed positions.
        
        Args:
            position (numpy.ndarray): Current position (x, y)
        """
        if not self.dynamic_bounds:
            return
            
        x, y = position[:2]
        self.maze_bounds['min_x'] = min(self.maze_bounds['min_x'], x)
        self.maze_bounds['max_x'] = max(self.maze_bounds['max_x'], x)
        self.maze_bounds['min_y'] = min(self.maze_bounds['min_y'], y)
        self.maze_bounds['max_y'] = max(self.maze_bounds['max_y'], y)
    
    def update_position(self, position, reward=0, done=False):
        """
        Update metrics based on current position.
        
        Args:
            position (numpy.ndarray): Current position (x, y)
            reward (float): Current reward
            done (bool): Whether episode is done
            
        Returns:
            dict: Current metrics
        """
        try:
            # Skip invalid positions
            if np.isnan(position).any() or np.isinf(position).any():
                return {}
                
            # Update bounds if dynamic
            self.update_bounds(position)
            
            # Convert to grid coordinates 
            grid_x, grid_y = self._position_to_grid(position)
            
            # Update visited cells
            self.visited_cells[grid_x, grid_y] = 1
            self.visit_counts[grid_x, grid_y] += 1
            
            # If this is the first position of an episode, record start position
            if len(self.episode_positions) == 0:
                self.starting_positions.append(position.copy())
            
            # Store current position
            self.episode_positions.append(position.copy())
            
            # Calculate distance from start
            if len(self.starting_positions) > 0:
                start_pos = self.starting_positions[-1]
                distance = np.linalg.norm(position[:2] - start_pos[:2])
                self.episode_max_distance = max(self.episode_max_distance, distance)
                self.max_distance = max(self.max_distance, distance)
            
            # Track goal distance if available
            if self.goal_position is not None:
                goal_distance = np.linalg.norm(position[:2] - self.goal_position[:2])
                self.episode_min_goal_distance = min(self.episode_min_goal_distance, goal_distance)
                self.min_goal_distance = min(self.min_goal_distance, goal_distance)
                
                # Count proximity to goal
                if goal_distance < 0.5:
                    self.goal_proximity_count += 1
                    self.episode_goal_reached = True
            
            # Generate heatmap at end of episode
            if done and self.log_dir and hasattr(self, 'episode_count'):
                if self.episode_count % self.heatmap_interval == 0:  # Every heatmap_interval episodes
                    self._generate_heatmap(self.episode_count)
            
            # Return current metrics
            return {
                'visit_count': int(np.sum(self.visit_counts)),
                'unique_cells': int(np.sum(self.visited_cells)),
                'max_distance': float(self.episode_max_distance),
                'min_goal_distance': float(self.episode_min_goal_distance),
                'goal_reached': int(self.episode_goal_reached)
            }
            
        except Exception as e:
            print(f"Error in update_position: {e}")
            return {}
    
    def end_episode(self, episode_count):
        """
        Call at the end of an episode to finalize metrics.
        
        Args:
            episode_count (int): Current episode number
            
        Returns:
            dict: Episode metrics
        """
        self.episode_count = episode_count
        coverage = self.get_coverage()
        visitation_entropy = self.get_state_visitation_entropy()
        
        return {
            'coverage': coverage,
            'max_distance': self.episode_max_distance,
            'min_goal_distance': self.episode_min_goal_distance,
            'goal_reached': int(self.episode_goal_reached),
            'goal_proximity_count': self.goal_proximity_count,
            'visitation_entropy': visitation_entropy,
            'episode_length': len(self.episode_positions)
        }
    
    def get_coverage(self):
        """
        Calculate maze coverage percentage.
        
        Returns:
            float: Percentage of maze covered
        """
        return np.sum(self.visited_cells) / (self.grid_size * self.grid_size)
    
    def get_state_visitation_entropy(self):
        """
        Calculate entropy of state visitation distribution.
        
        Returns:
            float: Entropy value
        """
        if np.sum(self.visit_counts) == 0:
            return 0.0
            
        # Convert visits to probability distribution
        visit_probs = self.visit_counts.flatten() + 1e-10  # Avoid zeros
        visit_probs = visit_probs / np.sum(visit_probs)
        entropy = -np.sum(visit_probs * np.log(visit_probs))
        return entropy
    
    def _position_to_grid(self, position):
        """
        Convert environment position to grid coordinates.
        
        Args:
            position (numpy.ndarray): Position in environment coordinates
            
        Returns:
            tuple: (grid_x, grid_y) indices
        """
        pos_x, pos_y = position[:2]  # Take only x,y coordinates
        
        # Calculate grid position based on current bounds
        grid_x = int((pos_x - self.maze_bounds['min_x']) / 
                     (self.maze_bounds['max_x'] - self.maze_bounds['min_x']) * 
                     self.grid_size)
        grid_y = int((pos_y - self.maze_bounds['min_y']) / 
                     (self.maze_bounds['max_y'] - self.maze_bounds['min_y']) * 
                     self.grid_size)
        
        # Ensure within bounds
        grid_x = max(0, min(grid_x, self.grid_size-1))
        grid_y = max(0, min(grid_y, self.grid_size-1)) 
        
        return grid_x, grid_y
    
    def _generate_heatmap(self, episode):
        """
        Generate visitation heatmap with maze wall overlay.
        
        Args:
            episode (int): Current episode number
        """
        try:
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
            
            # Create heatmap
            hm = sns.heatmap(
                self.visit_counts, 
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Visit Count', 'shrink': 0.8}
            )
            
            # Add grid overlay
            ax.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Remove numerical ticks from axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # If maze_map is available, overlay the walls
            if hasattr(self, 'maze_map') and self.maze_map is not None:
                maze_map = self.maze_map
                maze_height, maze_width = len(maze_map), len(maze_map[0])
                grid_height, grid_width = self.grid_size, self.grid_size
                
                # Scale factors for mapping maze coords to grid coords
                scale_y = grid_height / maze_height
                scale_x = grid_width / maze_width
                
                # Draw maze walls with thick black lines
                for y in range(maze_height):
                    for x in range(maze_width):
                        if maze_map[y][x] == 1:  # Wall
                            # Calculate corresponding grid cell coordinates
                            grid_y = int(y * scale_y)
                            grid_x = int(x * scale_x)
                            width = max(1, int(scale_x))
                            height = max(1, int(scale_y))
                            
                            # Add dark rectangle for wall
                            rect = plt.Rectangle(
                                (grid_x, grid_y), 
                                width, height, 
                                facecolor='black',
                                edgecolor='white',
                                linewidth=0.5,
                                alpha=0.8,
                                zorder=5  # Make sure walls are drawn on top of heatmap
                            )
                            ax.add_patch(rect)
            
            # If goal is available, mark it on the heatmap
            if self.goal_position is not None:
                goal_grid_x, goal_grid_y = self._position_to_grid(self.goal_position)
                ax.scatter(
                    goal_grid_y + 0.5,  # Center in cell
                    goal_grid_x + 0.5, 
                    c='red', 
                    marker='*', 
                    s=250,
                    edgecolor='black',
                    linewidth=0.8,
                    label='Goal',
                    zorder=10  # Make sure goal is on top of everything
                )
            
            # Mark starting position
            if self.starting_positions:
                start_x, start_y = self._position_to_grid(self.starting_positions[-1])
                ax.scatter(
                    start_y + 0.5, 
                    start_x + 0.5, 
                    c='blue', 
                    marker='o', 
                    s=180,
                    edgecolor='black',
                    linewidth=0.8,
                    label='Start',
                    zorder=10  # Make sure start is on top of everything
                )
                
            # Add legend
            plt.legend(
                loc='upper right',
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                fancybox=False,
                fontsize=10
            )
            
            # Add title
            plt.title(
                f"State Visitation Heatmap (Episode {episode})",
                fontsize=14,
                fontweight='bold',
                pad=10
            )
            
            # Apply tight layout with padding
            plt.tight_layout(pad=2.0)
            
            # Save figure
            save_path = os.path.join(self.heatmap_dir, f"heatmap_ep{episode}.png")
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5
            )
            
            # Close the figure to free memory
            plt.close(fig)
            
            print(f"Heatmap saved for episode {episode}")
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()


class RunningMeanStd:
    """Calculate running mean and standard deviation."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter:
    """Performs a forward filtering of rewards for normalization."""
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def make_train_data(reward, done, value, gamma, num_step, num_worker, use_gae=True, lam=0.95):
    """
    Calculate returns and advantages for PPO.
    
    Args:
        reward (numpy.ndarray): Rewards of shape [num_worker, num_step]
        done (numpy.ndarray): Done flags of shape [num_worker, num_step]
        value (numpy.ndarray): Values of shape [num_worker, num_step+1]
        gamma (float): Discount factor
        num_step (int): Number of steps per rollout
        num_worker (int): Number of parallel environments
        use_gae (bool): Whether to use GAE
        lam (float): GAE lambda parameter
        
    Returns:
        tuple: (discounted_return, advantage)
    """
    discounted_return = np.empty([num_worker, num_step])
    
    # Calculate discounted returns and advantages
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lam * (1 - done[:, t]) * gae
            discounted_return[:, t] = gae + value[:, t]

        # Calculate advantage
        advantage = discounted_return - value[:, :-1]
    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # Calculate advantage
        advantage = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), advantage.reshape([-1])


def check_path(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def setup_experiment_folder(exp_name=None):
    """
    Setup folders for the experiment with timestamped name.
    
    Args:
        exp_name (str): Optional experiment name prefix
        
    Returns:
        tuple: (log_dir, model_dir, experiment_name)
    """
    # Generate experiment name with timestamp
    timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
    if exp_name:
        experiment_name = f"{exp_name}_{timestamp}"
    else:
        experiment_name = f"DRND_PointMaze_{timestamp}"
    
    # Create necessary directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs', experiment_name)
    model_dir = os.path.join(script_dir, 'models', experiment_name)
    
    check_path(log_dir)
    check_path(model_dir)
    check_path(os.path.join(log_dir, 'heatmaps'))
    
    return log_dir, model_dir, experiment_name


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")


def save_config(config, log_dir):
    """
    Save experiment configuration to file.
    
    Args:
        config (dict): Configuration dictionary
        log_dir (str): Log directory
    """
    config_path = os.path.join(log_dir, 'config.json')
    
    # Convert non-serializable items to strings
    serializable_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            serializable_config[k] = v
        else:
            serializable_config[k] = str(v)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    print(f"Saved configuration to {config_path}")