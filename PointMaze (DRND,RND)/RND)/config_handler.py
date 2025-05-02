"""
Configuration handler for DRND experiments with different maze types.
This module reads the config.conf file and provides the appropriate maze configuration.
"""
import os
import configparser
import numpy as np
from maze_config import get_maze_config, list_available_mazes

def load_config(config_path=None):
    """
    Load configuration from config.conf file.
    
    Args:
        config_path (str): Path to config file, or None to use default path
        
    Returns:
        tuple: (config, maze_config) - ConfigParser object and maze configuration dictionary
    """
    # Use default path if not provided
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.conf')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Parse config file
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Check for MazeType parameter
    if 'MazeType' in config['DEFAULT']:
        maze_type = config['DEFAULT']['MazeType']
    else:
        # If MazeType is not specified, infer from EnvID
        env_id = config['DEFAULT']['EnvID']
        if 'UMaze' in env_id:
            maze_type = 'umaze'
        elif 'Open' in env_id:
            maze_type = 'open'
        elif 'Medium' in env_id:
            maze_type = 'medium'
        elif 'Large' in env_id:
            maze_type = 'large'
        else:
            # Default to UMaze if can't determine
            maze_type = 'umaze'
        
        # Add MazeType to config for consistency
        config['DEFAULT']['MazeType'] = maze_type
    
    # Get maze configuration
    maze_config = get_maze_config(maze_type)
    
    # Ensure EnvID matches the maze type
    if config['DEFAULT']['EnvID'] != maze_config['env_id']:
        print(f"Warning: EnvID in config ({config['DEFAULT']['EnvID']}) doesn't match maze type {maze_type}.")
        print(f"Setting EnvID to {maze_config['env_id']} based on MazeType parameter.")
        config['DEFAULT']['EnvID'] = maze_config['env_id']
    
    return config, maze_config

def update_config_file(config_path=None, maze_type=None):
    """
    Update the config.conf file with a new maze type.
    
    Args:
        config_path (str): Path to config file, or None to use default path
        maze_type (str): Maze type to set
        
    Returns:
        dict: Updated maze configuration
    """
    # Use default path if not provided
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.conf')
    
    # Check if maze type is valid
    if maze_type is not None and maze_type.lower() not in list_available_mazes():
        print(f"Warning: Invalid maze type '{maze_type}'.")
        print(f"Available maze types: {', '.join(list_available_mazes())}")
        return None
    
    # Load current configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if maze_type is not None:
        # Get maze configuration
        maze_config = get_maze_config(maze_type)
        
        # Update config
        config['DEFAULT']['MazeType'] = maze_type
        config['DEFAULT']['EnvID'] = maze_config['env_id']
        
        # Save changes
        with open(config_path, 'w') as configfile:
            config.write(configfile)
            
        print(f"Updated config.conf with MazeType={maze_type}, EnvID={maze_config['env_id']}")
        return maze_config
    
    return None

if __name__ == "__main__":
    # Example usage
    config, maze_config = load_config()
    print(f"Loaded configuration:")
    print(f"  MazeType: {config['DEFAULT'].get('MazeType', 'Not specified')}")
    print(f"  EnvID: {config['DEFAULT']['EnvID']}")
    print(f"  Fixed Start: {maze_config['fixed_start']}")
    print(f"  Fixed Goal: {maze_config['fixed_goal']}")