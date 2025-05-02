"""
Maze configuration module for PointMaze environments.
Defines different maze types and their corresponding fixed start/goal positions.
"""
import numpy as np

# Maze definitions with corrected positions
MAZE_CONFIGS = {
    # U-Maze configuration
    "umaze": {
        "env_id": "PointMaze_UMaze-v3",
        "maze_map": [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1], 
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        "fixed_start": np.array([-2.0, -2.0], dtype=np.float32),  # Bottom-left corridor
        "fixed_goal": np.array([2.0, -2.0], dtype=np.float32),     # Bottom-right corridor
        "description": "U-shaped maze with start at bottom-left and goal at bottom-right"
    },
    
    # Open maze configuration - fixed the start position
    "open": {
        "env_id": "PointMaze_Open-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ],
        "fixed_start": np.array([-2.0, 6.0], dtype=np.float32),   # Left side of open area (valid cell)
        "fixed_goal": np.array([3.0, -1.0], dtype=np.float32),     # Right side of open area
        "description": "Open maze with start at left and goal at right"
    },
    
    # Medium maze configuration - fixed start and goal positions
    "medium": {
        "env_id": "PointMaze_Medium-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        "fixed_start": np.array([-2.0, 8.0], dtype=np.float32),   # Top-left valid corridor
        "fixed_goal": np.array([8.0, -2.0], dtype=np.float32),    # Bottom-right valid corridor
        "description": "Medium maze with several corridors and rooms"
    },
    
    # Large maze configuration - fixed start and goal positions
    "large": {
        "env_id": "PointMaze_Large-v3",
        "maze_map": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        "fixed_start": np.array([-2.0, 16.0], dtype=np.float32),   # Top-left valid corridor
        "fixed_goal": np.array([10.0, -2.0], dtype=np.float32),    # Bottom-right valid corridor
        "description": "Large complex maze with multiple paths"
    }
}

def get_maze_config(maze_type="umaze"):
    """
    Get maze configuration for the specified maze type.
    
    Args:
        maze_type (str): One of 'umaze', 'open', 'medium', 'large'
        
    Returns:
        dict: Maze configuration or None if maze_type is invalid
    """
    if maze_type.lower() not in MAZE_CONFIGS:
        print(f"Warning: Invalid maze type '{maze_type}'. Using default 'umaze'.")
        maze_type = "umaze"
    
    return MAZE_CONFIGS[maze_type.lower()]

def list_available_mazes():
    """
    List all available maze configurations.
    
    Returns:
        list: List of available maze types
    """
    return list(MAZE_CONFIGS.keys())