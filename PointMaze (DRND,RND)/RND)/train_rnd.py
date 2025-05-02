import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from config import *
from rnd_model import RNDAgent  # Import our RND agent instead of DRND
from utils import (
    RunningMeanStd, RewardForwardFilter, make_train_data,
    set_random_seeds, setup_experiment_folder, save_config
)
from env_worker import create_parallel_envs

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

def main():
    # Load configuration
    env_id = default_config.get('EnvID')
    env_type = default_config.get('EnvType')
    num_worker = default_config.getint('NumEnv')
    num_step = default_config.getint('NumStep')
    max_episode_steps = default_config.getint('MaxStepPerEpisode')
    seed = default_config.getint('Seed')
    
    # Print the seed that will be used
    print(f"Using seed: {seed}")
    
    # Environment specific settings
    continuing_task = default_config.getboolean('ContinuingTask')
    reset_target = default_config.getboolean('ResetTarget')
    use_dense_reward = default_config.getboolean('UseDenseReward')
    
    # Create experiment directories with RND tag to differentiate from DRND
    log_dir, model_dir, experiment_name = setup_experiment_folder(f"RND_{env_type}_{env_id.split('-')[0]}")
    print(f"Experiment: {experiment_name}")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    
    # Save configuration to log directory
    # Access the config parser directly from the imported module
    from config import config
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    config_dict['DEFAULT'] = dict(config.items('DEFAULT'))
    save_config(config_dict, log_dir)
    
    # Set up tensorboard logging
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    from maze_config import get_maze_config

    # Get the maze type from config
    maze_type = default_config.get('MazeType', 'umaze')
    maze_config = get_maze_config(maze_type)

    # Use the goal and start positions from the maze configuration
    fixed_goal = maze_config['fixed_goal']
    fixed_start = maze_config['fixed_start']
    
    # Save the fixed goal and start positions to a file for reference
    positions_path = os.path.join(log_dir, 'fixed_positions.txt')
    with open(positions_path, 'w') as f:
        f.write(f"Fixed goal for all environments: {fixed_goal}\n")
        f.write(f"Fixed start for all environments: {fixed_start}\n")
        f.write(f"Using maze type: {maze_type}\n")
    
    # Get render environment setting
    render_env = default_config.getboolean('RenderEnv', fallback=False)
    
    # Create parallel environments
    parent_conns, workers, env_info = create_parallel_envs(
        env_id=env_id,
        num_envs=num_worker,
        log_dir=log_dir,
        fixed_goal=fixed_goal,
        fixed_start=fixed_start,
        seed=seed,
        continuing_task=continuing_task,
        reset_target=reset_target,
        use_dense_reward=use_dense_reward,
        max_episode_steps=max_episode_steps,
        maze_map=None,
        disable_rendering=not render_env
    )
    
    # Extract environment dimensions
    obs_dim = env_info['obs_dim']
    action_dim = env_info['action_dim']
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Set learning parameters
    use_gae = default_config.getboolean('UseGAE')
    use_gpu = default_config.getboolean('UseGPU') and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # Set model hyperparameters
    gamma = default_config.getfloat('Gamma')
    int_gamma = default_config.getfloat('IntGamma')
    lam = default_config.getfloat('Lambda')
    entropy_coef = default_config.getfloat('EntropyCoef')
    value_coef = default_config.getfloat('ValueCoef')
    ext_coef = default_config.getfloat('ExtCoef')
    int_coef = default_config.getfloat('IntCoef')
    num_epochs = default_config.getint('Epoch')
    num_mini_batch = default_config.getint('MiniBatch')
    ppo_eps = default_config.getfloat('PPOEps')
    max_grad_norm = default_config.getfloat('ClipGradNorm')
    lr = default_config.getfloat('LearningRate')
    rnd_lr = default_config.getfloat('DRNDLearningRate')  # Reuse DRND learning rate parameter
    update_proportion = default_config.getfloat('UpdateProportion')
    output_dim = 256

    # Print key settings
    print(f"Running on device: {device}")
    print(f"Number of environments: {num_worker}")
    print(f"Steps per environment: {num_step}")
    print(f"Total batch size: {num_worker * num_step}")
    print(f"Using GAE: {use_gae}")
    print(f"update_proportion: {update_proportion}")
    print(f"ext_coef: {ext_coef}")
    print(f"int_coef: {int_coef}")
    print(f"gamma: {gamma}")
    print(f"int_gamma: {int_gamma}")
    print(f"rnd_output_dim: {output_dim}")
    
    
    # Initialize RND Agent
    agent = RNDAgent(
        input_size=obs_dim,
        action_size=action_dim,
        device=device,
        lr=lr,
        drnd_lr=rnd_lr,
        gamma=gamma,
        gae_lambda=lam,
        ppo_epsilon=ppo_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        drnd_output_dim=output_dim  # Size of feature representation
    )
    
    # Save initial model
    model_path = os.path.join(model_dir, 'model_initial.pt')
    rnd_path = os.path.join(model_dir, 'rnd_initial.pt')
    torch.save(agent.model.state_dict(), model_path)
    torch.save(agent.rnd.state_dict(), rnd_path)
    
    # Initialize statistics
    obs_rms = RunningMeanStd(shape=(1, obs_dim))
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(int_gamma)
    
    # Initialize states for all environments
    states = np.zeros([num_worker, obs_dim])
    
    # Reset all environments to get initial states
    for worker_id, parent_conn in enumerate(parent_conns):
        parent_conn.send(('reset', None))
    
    for worker_id, parent_conn in enumerate(parent_conns):
        states[worker_id] = parent_conn.recv()
    
    # Training statistics
    global_update = 0
    global_step = 0
    
    # Per-episode statistics tracking
    episode_rewards = np.zeros(num_worker)
    episode_steps = np.zeros(num_worker)
    episode_intrinsic_rewards = np.zeros(num_worker)
    episode_counts = np.zeros(num_worker)
    
    # Initialize observation normalization parameters
    print('Initializing observation normalization parameters...')
    # We'll use early training steps to normalize observations
    obs_norm_step = default_config.getint('ObsNormStep')
    next_obs_buffer = []
    
    for step in range(obs_norm_step):
        # Sample random actions
        actions = np.random.uniform(-1, 1, size=(num_worker, action_dim))
        
        # Step environments with random actions
        for worker_id, parent_conn in enumerate(parent_conns):
            parent_conn.send(('step', actions[worker_id]))
        
        # Collect observations
        for worker_id, parent_conn in enumerate(parent_conns):
            s, r, d, rd, lr, next_obs, _ = parent_conn.recv()
            next_obs_buffer.append(next_obs)
            
            # Reset environment if episode ended
            if d:
                parent_conn.send(('reset', None))
                states[worker_id] = parent_conn.recv()
        
        # Update observation normalization
        if len(next_obs_buffer) >= 256:
            next_obs_batch = np.concatenate(next_obs_buffer)
            obs_rms.update(next_obs_batch)
            next_obs_buffer = []
    
    print('Observation normalization initialized!')
    
    # Main training loop (2 million steps)
    while global_step < 2e6:
        # Accumulate data over multiple environment steps
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        log_probs_buffer = []
        values_ext_buffer = []
        values_int_buffer = []
        next_states_buffer = []
        intrinsic_rewards_buffer = []
        
        global_step += (num_worker * num_step)
        global_update += 1
        
        # Step 1: n-step rollout
        for step in range(num_step):
            # Normalize observations
            norm_states = np.clip((states - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5)
            
            # Get actions from policy
            actions, log_probs, values_ext, values_int, entropy, policy_params = agent.get_action(norm_states)
            
            # Step environments
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(('step', actions[worker_id]))
            
            # Collect results
            states_buffer.append(states)
            actions_buffer.append(actions)
            log_probs_buffer.append(log_probs)
            values_ext_buffer.append(values_ext)
            values_int_buffer.append(values_int)
            
            next_states = []
            rewards = []
            dones = []
            env_infos = []
            next_obs_batch = []
            
            # Process environment results
            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, real_done, log_reward, next_obs, info = parent_conn.recv()
                
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                next_obs_batch.append(next_obs)
                env_infos.append(info)
                
                # Update episode tracking
                episode_rewards[worker_id] += reward
                episode_steps[worker_id] += 1
                
                # If episode ended, log results and reset tracking
                if done:
                    episode_counts[worker_id] += 1
                    
                    # Log episode results
                    writer.add_scalar(f'rewards/episode_reward', episode_rewards[worker_id], global_step)
                    writer.add_scalar(f'rewards/episode_length', episode_steps[worker_id], global_step)
                    writer.add_scalar(f'rewards/intrinsic_reward', episode_intrinsic_rewards[worker_id], global_step)
                    
                    # Log environment metrics
                    if info:
                        for k, v in info.items():
                            writer.add_scalar(f'env{worker_id}/{k}', v, global_step)
                    
                    # Reset episode tracking
                    episode_rewards[worker_id] = 0
                    episode_steps[worker_id] = 0
                    episode_intrinsic_rewards[worker_id] = 0
            
            # Update states
            states = np.vstack(next_states)
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            next_obs_batch = np.vstack(next_obs_batch)
            
            # Calculate intrinsic rewards using RND
            intrinsic_reward = agent.compute_intrinsic_reward(
                np.clip((next_obs_batch - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5)
            )
            
            # Track buffers
            rewards_buffer.append(rewards)
            dones_buffer.append(dones)
            next_states_buffer.append(states)
            intrinsic_rewards_buffer.append(intrinsic_reward)
            
            # Update episode intrinsic rewards
            for worker_id, int_rew in enumerate(intrinsic_reward):
                episode_intrinsic_rewards[worker_id] += int_rew
        
        # Get final values for bootstrapping
        norm_states = np.clip((states - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5)
        _, _, next_values_ext, next_values_int, _, _ = agent.get_action(norm_states)
        
        # Process collected rollout data
        states_buffer = np.array(states_buffer).reshape(-1, obs_dim)
        actions_buffer = np.array(actions_buffer).reshape(-1, action_dim)
        log_probs_buffer = np.array(log_probs_buffer).flatten()
        rewards_buffer = np.array(rewards_buffer).T
        dones_buffer = np.array(dones_buffer).T
        values_ext_buffer = np.array(values_ext_buffer).T
        values_int_buffer = np.array(values_int_buffer).T
        next_states_buffer = np.array(next_states_buffer).reshape(-1, obs_dim)
        intrinsic_rewards_buffer = np.array(intrinsic_rewards_buffer).T
        
        # Normalize intrinsic rewards using running statistics
        total_int_reward_per_env = np.array([discounted_reward.update(reward_per_step) 
                                           for reward_per_step in intrinsic_rewards_buffer.T])
        mean, std, count = np.mean(total_int_reward_per_env), np.std(total_int_reward_per_env), len(total_int_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)
        
        # Apply normalization to intrinsic rewards
        intrinsic_rewards_buffer /= np.sqrt(reward_rms.var)
        
        # Log average rewards
        writer.add_scalar('rewards/extrinsic', np.mean(rewards_buffer), global_step)
        writer.add_scalar('rewards/intrinsic', np.mean(intrinsic_rewards_buffer), global_step)
        
        # Prepare value targets and advantages
        # Concatenate values with bootstrapped next values
        values_ext = np.concatenate([values_ext_buffer, np.expand_dims(next_values_ext, 1)], axis=1)
        values_int = np.concatenate([values_int_buffer, np.expand_dims(next_values_int, 1)], axis=1)
        
        # Calculate advantages and returns
        # Extrinsic returns and advantages
        ext_returns, ext_advantages = make_train_data(
            rewards_buffer,
            dones_buffer,
            values_ext,
            gamma,
            num_step,
            num_worker,
            use_gae=use_gae,
            lam=lam
        )
        
        # Intrinsic returns and advantages (non-episodic)
        # For intrinsic rewards, we use a non-episodic approach
        int_returns, int_advantages = make_train_data(
            intrinsic_rewards_buffer,
            np.zeros_like(dones_buffer),  # Treating as non-episodic
            values_int,
            int_gamma,
            num_step,
            num_worker,
            use_gae=use_gae,
            lam=lam
        )
        
        # Combine advantages with coefficients
        total_advantages = ext_advantages * ext_coef + int_advantages * int_coef
        
        # Update observation normalization with latest batch
        obs_rms.update(next_states_buffer)
        
        # Prepare state normalization for training
        norm_states_buffer = np.clip((states_buffer - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5)
        norm_next_states_buffer = np.clip((next_states_buffer - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -5, 5)
        
        # Training step
        # Convert numpy arrays to torch tensors
        states_tensor = torch.FloatTensor(norm_states_buffer).to(device)
        actions_tensor = torch.FloatTensor(actions_buffer).to(device)
        old_log_probs_tensor = torch.FloatTensor(log_probs_buffer).to(device)
        ext_returns_tensor = torch.FloatTensor(ext_returns).to(device)
        int_returns_tensor = torch.FloatTensor(int_returns).to(device)
        advantages_tensor = torch.FloatTensor(total_advantages).to(device)
        next_states_tensor = torch.FloatTensor(norm_next_states_buffer).to(device)
        
        # Update RND predictor network
        rnd_loss = agent.update_rnd(next_states_tensor, update_proportion)
        
        # Log RND loss
        writer.add_scalar('training/prediction_error', rnd_loss, global_step)
        
        # Calculate batch size for PPO updates
        batch_size = num_worker * num_step
        mini_batch_size = batch_size // num_mini_batch
        
        # PPO update loop
        for epoch in range(num_epochs):
            # Shuffle the data
            indices = np.random.permutation(batch_size)
            
            # Update in mini-batches
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = indices[start:end]
                
                # Update policy and value networks
                policy_loss, value_loss, entropy = agent.update_policy(
                    states_tensor[mini_batch_indices],
                    actions_tensor[mini_batch_indices],
                    old_log_probs_tensor[mini_batch_indices],
                    ext_returns_tensor[mini_batch_indices],
                    int_returns_tensor[mini_batch_indices],
                    advantages_tensor[mini_batch_indices]
                )
                
                # Log PPO update metrics
                writer.add_scalar('training/policy_loss', policy_loss, global_step + start)
                writer.add_scalar('training/value_loss', value_loss, global_step + start)
                writer.add_scalar('training/entropy', entropy, global_step + start)
        
        # Save model periodically
        if global_update % 100 == 0:
            print(f"Saving model after {global_step} steps, update {global_update}")
            model_save_path = os.path.join(model_dir, f'model_{global_update}.pt')
            rnd_save_path = os.path.join(model_dir, f'rnd_{global_update}.pt')
            torch.save(agent.model.state_dict(), model_save_path)
            torch.save(agent.rnd.state_dict(), rnd_save_path)
            
            # Save latest model
            latest_model_path = os.path.join(model_dir, 'model_latest.pt')
            latest_rnd_path = os.path.join(model_dir, 'rnd_latest.pt')
            torch.save(agent.model.state_dict(), latest_model_path)
            torch.save(agent.rnd.state_dict(), latest_rnd_path)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during training: {e}")