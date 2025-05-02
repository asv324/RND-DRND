import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper
import os

class PointMazeWrapper(Wrapper):
    """
    A wrapper for PointMaze environments that ensures goal consistency
    and adds functionality for tracking visitation statistics.
    """
    def __init__(self, 
                 env_id, 
                 fixed_goal=None, 
                 fixed_start=None, 
                 env_idx=0,
                 render_mode=None,
                 maze_map=None,
                 max_episode_steps=None,
                 continuing_task=True,
                 reset_target=False):
        """
        Initialize the PointMaze wrapper.
        
        Args:
            env_id (str): The environment ID to create
            fixed_goal (numpy.ndarray): Optional fixed goal position (2D array [x, y])
            fixed_start (numpy.ndarray): Optional fixed start position (2D array [x, y])
            env_idx (int): Environment index for multiprocessing
            render_mode (str): Optional render mode
            maze_map (list): Optional custom maze map
            max_episode_steps (int): Max steps per episode
            continuing_task (bool): Whether to make the task continuing
            reset_target (bool): Whether to reset target when reached
        """
        # Create the base environment
        env_kwargs = {'continuing_task': continuing_task, 'reset_target': reset_target}
        if maze_map is not None:
            env_kwargs['maze_map'] = maze_map
        if max_episode_steps is not None:
            env_kwargs['max_episode_steps'] = max_episode_steps
        if render_mode is not None:
            env_kwargs['render_mode'] = render_mode
        
        env = gym.make(env_id, **env_kwargs)
        
        super().__init__(env)
        self.env_idx = env_idx
        self._fixed_goal = fixed_goal
        self._fixed_start = fixed_start
        self._original_goal = None
        self.last_xy_position = None
        self.episode_positions = []
        self.maze_map = maze_map  # Store maze_map for later use
        
        # Convert fixed coordinates to cell indices if provided
        self._fixed_goal_cell = None
        self._fixed_start_cell = None
        
        # The default maze scaling in PointMaze is 2.0
        # Each cell is 2x2 units in the continuous space
        maze_scaling = 2.0
        
        # If fixed goal is provided in continuous coordinates, convert to cell indices
        if self._fixed_goal is not None:
            # Add 4 to get positive indices (maze is centered around origin)
            # Then divide by 2 for the cell size
            self._fixed_goal_cell = np.array([
                int((self._fixed_goal[0] + 4) / maze_scaling),
                int((self._fixed_goal[1] + 4) / maze_scaling)
            ], dtype=np.int32)
            print(f"[Env {env_idx}] Fixed goal position {self._fixed_goal} converted to cell indices {self._fixed_goal_cell}")
        
        # If fixed start is provided in continuous coordinates, convert to cell indices
        if self._fixed_start is not None:
            self._fixed_start_cell = np.array([
                int((self._fixed_start[0] + 4) / maze_scaling),
                int((self._fixed_start[1] + 4) / maze_scaling)
            ], dtype=np.int32)
            print(f"[Env {env_idx}] Fixed start position {self._fixed_start} converted to cell indices {self._fixed_start_cell}")
        
        # Properties for tracking visitation
        self.visited_positions = []
        self.total_visits = []
        self.successful_episodes = 0
        self.episode_count = 0
        
        # Try to get the maze layout from the environment
        if hasattr(self.env.unwrapped, 'maze_map') and self.maze_map is None:
            self.maze_map = self.env.unwrapped.maze_map
            print(f"[Env {env_idx}] Retrieved maze map from environment")
        
        print(f"[Env {env_idx}] PointMaze wrapper initialized with continuing_task={continuing_task}, "
              f"reset_target={reset_target}")
    
    def get_normalized_position(self, observation):
        """
        Get the normalized (x, y) position from an observation.
        
        Args:
            observation: The observation dictionary
            
        Returns:
            tuple: (x, y) position coordinates
        """
        # Extract current position from observation
        if isinstance(observation, dict) and 'observation' in observation:
            # 'observation' key with the first two elements being x, y position
            return observation['observation'][:2]
        else:
            # Assume first two elements are x, y position
            return observation[:2]
    
    def get_desired_goal(self, observation):
        """
        Get the desired goal position from observation.
        
        Args:
            observation: The observation dictionary
            
        Returns:
            numpy.ndarray: Goal position (x, y)
        """
        if isinstance(observation, dict) and 'desired_goal' in observation:
            return observation['desired_goal']
        return self._fixed_goal  # Fallback to stored fixed goal
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment with fixed goal and start positions if specified.
        
        Args:
            seed: Random seed
            options: Optional reset options
            
        Returns:
            observation: First observation
            info: Information dictionary
        """
        # Initialize reset options
        reset_options = {}
        
        # Add fixed goal cell if specified
        if self._fixed_goal_cell is not None:
            reset_options['goal_cell'] = self._fixed_goal_cell.copy()
        
        # Add fixed start cell if specified
        if self._fixed_start_cell is not None:
            reset_options['reset_cell'] = self._fixed_start_cell.copy()
        
        # Combine with any options passed in
        if options is not None:
            reset_options.update(options)
        
        # Reset the environment with our options and seed
        obs, info = self.env.reset(seed=seed, options=reset_options if reset_options else None)
        
        # Store original goal for reference
        if self._original_goal is None and isinstance(obs, dict) and 'desired_goal' in obs:
            self._original_goal = obs['desired_goal'].copy()
            
            # If no fixed goal specified, use the original goal
            if self._fixed_goal is None:
                self._fixed_goal = self._original_goal.copy()
                print(f"[Env {self.env_idx}] Setting fixed goal to {self._fixed_goal}")
        
        # Initialize episode tracking
        self.last_xy_position = self.get_normalized_position(obs)
        self.episode_positions = [self.last_xy_position.copy()]
        self.episode_count += 1
        
        return obs, info
    
    def step(self, action):
        """
        Step the environment and track visitation statistics.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward received
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Take step in environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update position tracking
        curr_xy_position = self.get_normalized_position(obs)
        
        # Skip tracking if position is NaN
        if not np.isnan(curr_xy_position).any():
            # Track visitation
            self.episode_positions.append(curr_xy_position.copy())
            self.last_xy_position = curr_xy_position
            
            # If episode is done, update overall stats
            if terminated or truncated:
                if reward > 0:  # Successful completion
                    self.successful_episodes += 1
                
                # Store all positions visited in this episode
                self.visited_positions.extend(self.episode_positions)
                self.total_visits.append(len(self.episode_positions))
        
        return obs, reward, terminated, truncated, info
    
    def get_visitation_stats(self):
        """
        Get statistics about environment visitation.
        
        Returns:
            dict: Statistics dictionary
        """
        visit_count = len(self.visited_positions)
        unique_visit_count = len(np.unique(np.round(self.visited_positions, decimals=1), axis=0))
        success_rate = self.successful_episodes / max(1, self.episode_count)
        
        # Calculate maze coverage
        maze_bounds = self._get_maze_bounds()
        grid_size = 20  # Discretize maze into 20x20 grid for coverage calculation
        coverage_grid = np.zeros((grid_size, grid_size))
        
        for pos in self.visited_positions:
            # Skip if position is out of detected maze bounds
            if (pos[0] < maze_bounds['min_x'] or pos[0] > maze_bounds['max_x'] or
                pos[1] < maze_bounds['min_y'] or pos[1] > maze_bounds['max_y']):
                continue
            
            # Convert position to grid cell
            grid_x = int((pos[0] - maze_bounds['min_x']) / (maze_bounds['max_x'] - maze_bounds['min_x']) * (grid_size - 1))
            grid_y = int((pos[1] - maze_bounds['min_y']) / (maze_bounds['max_y'] - maze_bounds['min_y']) * (grid_size - 1))
            
            # Ensure within grid bounds
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))
            
            coverage_grid[grid_x, grid_y] = 1
        
        coverage = np.sum(coverage_grid) / (grid_size * grid_size)
        
        return {
            'visit_count': visit_count,
            'unique_visits': unique_visit_count,
            'success_rate': success_rate,
            'coverage': coverage,
            'episode_count': self.episode_count,
            'successful_episodes': self.successful_episodes,
            'maze_bounds': maze_bounds
        }
    
    def _get_maze_bounds(self):
        """
        Get the bounds of the maze based on visited positions.
        
        Returns:
            dict: Maze boundaries
        """
        if not self.visited_positions:
            # Default bounds if no positions tracked yet
            return {'min_x': -4, 'max_x': 4, 'min_y': -4, 'max_y': 4}
        
        # Convert to numpy array for easier manipulation
        positions = np.array(self.visited_positions)
        
        # Get min and max x, y coordinates with some padding
        min_x = np.min(positions[:, 0]) - 1
        max_x = np.max(positions[:, 0]) + 1
        min_y = np.min(positions[:, 1]) - 1
        max_y = np.max(positions[:, 1]) + 1
        
        return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}


def create_point_maze_env(
    env_id="PointMaze_UMaze-v3",
    fixed_goal=None,
    fixed_start=None,
    env_idx=0,
    render_mode=None,
    max_episode_steps=500,
    continuing_task=True,
    reset_target=False,
    use_dense_reward=False,
    maze_map=None
):
    """
    Create a PointMaze environment with the wrapper for consistent goals
    and visitation tracking.
    
    Args:
        env_id (str): Base environment ID
        fixed_goal (numpy.ndarray): Optional fixed goal position
        fixed_start (numpy.ndarray): Optional fixed start position
        env_idx (int): Environment index
        render_mode (str): Optional render mode
        max_episode_steps (int): Maximum steps per episode
        continuing_task (bool): Whether to make task continuing
        reset_target (bool): Whether to reset target when reached
        use_dense_reward (bool): Whether to use dense rewards
        maze_map (list): Optional custom maze map
        
    Returns:
        PointMazeWrapper: Wrapped environment
    """
    # Modify environment ID for dense reward if requested
    if use_dense_reward and 'Dense' not in env_id:
        # Split the ID to insert 'Dense' before the version number
        parts = env_id.split('-')
        base_id = parts[0]
        version = parts[1]
        env_id = f"{base_id}Dense-{version}"
    
    # Create and return wrapped environment
    return PointMazeWrapper(
        env_id=env_id,
        fixed_goal=fixed_goal,
        fixed_start=fixed_start,
        env_idx=env_idx,
        render_mode=render_mode,
        maze_map=maze_map,
        max_episode_steps=max_episode_steps,
        continuing_task=continuing_task,
        reset_target=reset_target
    )