"""
Training script for RND with different maze configurations.
This script wraps the original train_rnd.py to inject maze configurations.
"""
import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
import sys
import argparse

# Import custom modules
from config_handler import load_config

# Register gymnasium-robotics environments if needed
gym.register_envs(gymnasium_robotics)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RND on maze environments')
    parser.add_argument('--seed', type=int, help='Random seed for the experiment')
    parser.add_argument('--maze_type', type=str, help='Type of maze: umaze, open, medium, large')
    parser.add_argument('--num_env', type=int, help='Number of parallel environments')
    parser.add_argument('--name', type=str, help='Custom name for the experiment (used for log folders)')
    parser.add_argument('--intcoef', type=float, help='Intrinsic coefficient for the reward function')
    parser.add_argument('--extcoef', type=float, help='Extrinsic coefficient for the reward function')
    parser.add_argument('--gamma', type=float, help='Extrinsic Gamma')
    parser.add_argument('--intgamma', type=float, help='Intrinsic Gamma')
    parser.add_argument('--updateproportion', type=float, help='Update Proportion')
    args = parser.parse_args()
    
    # Load configuration and get maze config
    config, maze_config = load_config()
    
    # Override config with command line arguments if provided
    if args.seed is not None:
        config['DEFAULT']['Seed'] = str(args.seed)
        print(f"Overriding seed with command line value: {args.seed}")
    
    if args.maze_type is not None:
        config['DEFAULT']['MazeType'] = args.maze_type
        from maze_config import get_maze_config
        maze_config = get_maze_config(args.maze_type)
        config['DEFAULT']['EnvID'] = maze_config['env_id']
        print(f"Overriding maze type with command line value: {args.maze_type}")
    
    if args.num_env is not None:
        config['DEFAULT']['NumEnv'] = str(args.num_env)
        print(f"Overriding number of environments with command line value: {args.num_env}")

    if args.intcoef is not None:
        config['DEFAULT']['IntCoef'] = str(args.intcoef)
        print(f"Overriding intrinsic coefficient with command line value: {args.intcoef}")

    if args.extcoef is not None:
        config['DEFAULT']['ExtCoef'] = str(args.extcoef)
        print(f"Overriding extrinsic coefficient with command line value: {args.extcoef}")

    if args.gamma is not None:
        config['DEFAULT']['Gamma'] = str(args.gamma)
        print(f"Overriding extrinsic gamma with command line value: {args.gamma}")

    if args.intgamma is not None:
        config['DEFAULT']['IntGamma'] = str(args.intgamma)
        print(f"Overriding intrinsic gamma with command line value: {args.intgamma}")

    if args.updateproportion is not None:
        config['DEFAULT']['UpdateProportion'] = str(args.updateproportion)
        print(f"Overriding update proportion with command line value: {args.updateproportion}")
    
    # Save custom experiment name if provided
    custom_exp_name = args.name
    if custom_exp_name:
        print(f"Using custom experiment name: {custom_exp_name}")

    # Print maze configuration details
    maze_type = config['DEFAULT']['MazeType']
    env_id = config['DEFAULT']['EnvID']
    print(f"\n{'='*50}")
    print(f"Starting RND experiment with:")
    print(f"  Maze Type: {maze_type}")
    print(f"  Environment ID: {env_id}")
    print(f"  Seed: {config['DEFAULT']['Seed']}")
    print(f"  Description: {maze_config['description']}")
    print(f"  Start Position: {maze_config['fixed_start']}")
    print(f"  Goal Position: {maze_config['fixed_goal']}")
    print(f"{'='*50}\n")
    
    # IMPORTANT: Execute the main function from train_rnd.py but override the hardcoded positions
    # and update the config values before importing train_rnd.py
    
    # First, update the config.py's configuration values
    import configparser
    import os
    
    # Get the current directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.conf')
    
    # Save the modified configuration to the config file
    with open(config_path, 'w') as configfile:
        # Create a new ConfigParser object
        config_writer = configparser.ConfigParser()
        # Update with our modified config
        for section in config:
            if section not in config_writer:
                config_writer[section] = {}
            for key, value in config[section].items():
                config_writer[section][key] = value
        # Write to the file
        config_writer.write(configfile)
    
    print(f"Updated config.conf with new settings before importing train_rnd.py")

    # Override the setup_experiment_folder function before it's imported
    # We need to do this before importing train.py
    from utils import setup_experiment_folder as orig_setup_experiment_folder
    
    # Create a custom version that uses our experiment name
    def custom_setup_experiment_folder(exp_name=None):
        """Custom version of setup_experiment_folder that uses our provided name"""
        if custom_exp_name is not None:
            # Use the name provided in command line args
            timestamp = datetime.now().strftime("%m%d_%H-%M-%S")
            experiment_name = f"{custom_exp_name}_{timestamp}"
            
            # Create necessary directories
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, 'logs', experiment_name)
            model_dir = os.path.join(script_dir, 'models', experiment_name)
            
            # Create directories
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'heatmaps'), exist_ok=True)
            
            print(f"Created custom experiment directories with name: {experiment_name}")
            return log_dir, model_dir, experiment_name
        else:
            # Use the original implementation
            return orig_setup_experiment_folder(exp_name)
    
    # Now patch the function in the utils module
    import utils
    utils.setup_experiment_folder = custom_setup_experiment_folder
    
    # Now import train_rnd and the rest of the modules, which will read the updated config
    import train_rnd
    from env_worker import create_parallel_envs
    
    # Save original function
    original_create_parallel_envs = create_parallel_envs
    
    # Patch the create_parallel_envs function to inject maze configuration
    def patched_create_parallel_envs(env_id, num_envs, log_dir=None, fixed_goal=None, fixed_start=None, seed=None, 
                          continuing_task=True, reset_target=False, use_dense_reward=False, 
                          max_episode_steps=500, maze_map=None, disable_rendering=True):
        """Override the create_parallel_envs function to use maze configuration values"""
        print(f"Creating environments with maze_type={maze_type}")
        print(f"Using fixed start: {maze_config['fixed_start']}")
        print(f"Using fixed goal: {maze_config['fixed_goal']}")
        
        # Make sure to pass the correct seed value from the updated config
        actual_seed = int(config['DEFAULT']['Seed'])
        print(f"Using seed from config: {actual_seed}")
        
        return original_create_parallel_envs(
            env_id=env_id,
            num_envs=num_envs,
            log_dir=log_dir,
            fixed_goal=maze_config['fixed_goal'],  # Use fixed goal from maze config
            fixed_start=maze_config['fixed_start'],  # Use fixed start from maze config
            seed=actual_seed,  # Use the seed from the updated config
            continuing_task=continuing_task,
            reset_target=reset_target,
            use_dense_reward=use_dense_reward,
            max_episode_steps=max_episode_steps,
            maze_map=maze_config['maze_map'],  # Use maze map from maze config
            disable_rendering=disable_rendering
        )
    
    # Replace the original function with our patched version
    import env_worker
    env_worker.create_parallel_envs = patched_create_parallel_envs
    
    # Monkey patch the train_rnd.main function to skip the hardcoded fixed positions
    original_main = train_rnd.main
    
    def patched_main():
        # Override the fixed_goal and fixed_start in train module
        # This prevents the hardcoded values from being used
        train_rnd.fixed_goal = maze_config['fixed_goal']
        train_rnd.fixed_start = maze_config['fixed_start']
        
        # Make sure train.py uses the correct seed
        # We need to update the default_config in train_rnd.py as well 
        from config import default_config
        
        # Force reload the config module to get the updated values
        import importlib
        import config
        importlib.reload(config)
        
        # Now run the original main function
        return original_main()
    
    # Replace the original main function
    train_rnd.main = patched_main
    
    # Run the patched main function
    train_rnd.main()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during execution: {e}")