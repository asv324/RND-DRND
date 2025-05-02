import numpy as np
from torch.multiprocessing import Process, Pipe
from collections import deque
import gymnasium as gym
import gymnasium_robotics
import time

# Import the wrapper for point maze environments
from point_maze_wrapper import create_point_maze_env
from utils import ExplorationMetrics

# Make sure robotics environments are registered
gym.register_envs(gymnasium_robotics)

class WorkerProcess(Process):
    """
    Worker process for parallel environment execution.
    """
    def __init__(
        self,
        env_id,
        is_render,
        env_idx,
        child_conn,
        seed=None,
        max_episode_steps=500,
        fixed_goal=None,
        fixed_start=None,  # Added fixed_start parameter
        log_dir=None,
        continuing_task=True,
        reset_target=False,
        use_dense_reward=False,
        maze_map=None
    ):
        super(WorkerProcess, self).__init__()
        self.daemon = True
        
        # Store parameters
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.seed_value = seed
        self.max_episode_steps = max_episode_steps
        self.fixed_goal = fixed_goal
        self.fixed_start = fixed_start  # Store fixed_start
        self.log_dir = log_dir
        self.continuing_task = continuing_task
        self.reset_target = reset_target
        self.use_dense_reward = use_dense_reward
        self.maze_map = maze_map
        
        # Initialize tracking variables
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.recent_rew_list = deque(maxlen=100)
        
    def run(self):
        """
        Main process loop
        """
        try:
            # Create the environment with wrapper
            render_mode = "human" if self.is_render else None
            
            self.env = create_point_maze_env(
                env_id=self.env_id,
                fixed_goal=self.fixed_goal,
                fixed_start=self.fixed_start,  # Pass fixed_start
                env_idx=self.env_idx,
                render_mode=render_mode,
                max_episode_steps=self.max_episode_steps,
                continuing_task=self.continuing_task,
                reset_target=self.reset_target,
                use_dense_reward=self.use_dense_reward,
                maze_map=self.maze_map
            )
            
            # Set seed through reset (new Gymnasium API)
            effective_seed = self.seed_value + self.env_idx if self.seed_value is not None else None
            obs_dict, _ = self.env.reset(seed=effective_seed)
            
            # Extract observation and process for state
            if isinstance(obs_dict, dict) and 'observation' in obs_dict:
                state = obs_dict['observation']
            else:
                state = obs_dict
                
            if hasattr(self.env.unwrapped, 'maze_map') and self.maze_map is None:
                self.maze_map = self.env.unwrapped.maze_map
                print(f"[Env {self.env_idx}] Retrieved maze map from environment")
            
            # Set up exploration metrics
            self.exploration_metrics = ExplorationMetrics(
                grid_size=30,
                maze_bounds=None,  # Will be determined dynamically
                log_dir=self.log_dir,
                maze_map=self.maze_map  # Pass maze_map to ExplorationMetrics
            )
            
            # Set goal for exploration metrics if available
            if isinstance(obs_dict, dict) and 'desired_goal' in obs_dict:
                self.exploration_metrics.goal_position = obs_dict['desired_goal']
                print(f"[Env {self.env_idx}] Set goal for exploration metrics: {self.exploration_metrics.goal_position}")
            
            # Main loop
            while True:
                # Wait to receive action from main process
                cmd, data = self.child_conn.recv()
                
                if cmd == 'step':
                    # Unpack action
                    action = data
                    
                    # Execute action
                    obs_dict, reward, terminated, truncated, info = self.env.step(action)
                    
                    # Extract state and goal position
                    if isinstance(obs_dict, dict) and 'observation' in obs_dict:
                        next_state = obs_dict['observation']
                        # Update goal position for metrics
                        if 'desired_goal' in obs_dict:
                            self.exploration_metrics.goal_position = obs_dict['desired_goal']
                    else:
                        next_state = obs_dict
                    
                    # Determine if episode is done
                    done = terminated or truncated
                    
                    # Update reward tracking
                    self.rall += reward
                    self.steps += 1
                    
                    # Update exploration metrics
                    self.exploration_metrics.update_position(
                        np.array(next_state[:2]),  # Position components (x, y)
                        reward=reward,
                        done=done
                    )
                    
                    # Handle episode end
                    if done:
                        # Get exploration metrics
                        exp_metrics = self.exploration_metrics.end_episode(self.episode)
                        
                        self.recent_rlist.append(self.rall)
                        self.recent_rew_list.append(reward)
                        
                        print(f"[Episode {self.episode}({self.env_idx})] "
                            f"Step: {self.steps} "
                            f"Reward: {self.rall:.3f} "
                            f"Recent Avg Reward: {np.mean(self.recent_rlist):.3f} "
                            f"Coverage: {exp_metrics['coverage']:.4f}")
                        
                        self.steps = 0
                        self.episode += 1
                        self.rall = 0
                        
                        # Reset the environment with seed based on episode number
                        episode_seed = self.seed_value + self.env_idx + self.episode if self.seed_value else None
                        obs_dict, _ = self.env.reset(seed=episode_seed)
                        
                        # Extract observation
                        if isinstance(obs_dict, dict) and 'observation' in obs_dict:
                            next_state = obs_dict['observation']
                            # Update goal position for metrics
                            if 'desired_goal' in obs_dict:
                                self.exploration_metrics.goal_position = obs_dict['desired_goal']
                        else:
                            next_state = obs_dict
                        
                        # Reset exploration metrics for new episode
                        self.exploration_metrics.reset()
                    
                    # Send results back to parent
                    self.child_conn.send([
                        next_state,         # observation
                        reward,             # reward
                        done,               # done
                        done,               # real_done (same as done)
                        reward,             # log_reward
                        next_state.reshape(1, -1),  # next_obs for DRND
                        exp_metrics if done else {}  # exploration metrics
                    ])
                    
                    # Update state
                    state = next_state
                    
                elif cmd == 'reset':
                    # Reset the environment with seed
                    episode_seed = self.seed_value + self.env_idx + self.episode if self.seed_value else None
                    obs_dict, _ = self.env.reset(seed=episode_seed)
                    
                    # Extract observation
                    if isinstance(obs_dict, dict) and 'observation' in obs_dict:
                        state = obs_dict['observation']
                        # Update goal position for metrics
                        if 'desired_goal' in obs_dict:
                            self.exploration_metrics.goal_position = obs_dict['desired_goal']
                    else:
                        state = obs_dict
                    
                    # Reset episode stats
                    self.steps = 0
                    self.rall = 0
                    self.exploration_metrics.reset()
                    
                    # Send state back
                    self.child_conn.send(state)
                    
                elif cmd == 'close':
                    self.env.close()
                    self.child_conn.close()
                    break
                    
                elif cmd == 'get_env_info':
                    # Send back info about the environment
                    env_info = {
                        'observation_space': self.env.observation_space,
                        'action_space': self.env.action_space,
                    }
                    
                    # Extract spaces
                    if isinstance(self.env.observation_space, gym.spaces.Dict):
                        if 'observation' in self.env.observation_space.spaces:
                            env_info['obs_dim'] = self.env.observation_space.spaces['observation'].shape[0]
                        else:
                            env_info['obs_dim'] = sum(space.shape[0] for space in self.env.observation_space.spaces.values())
                    else:
                        env_info['obs_dim'] = self.env.observation_space.shape[0]
                    
                    env_info['action_dim'] = self.env.action_space.shape[0]
                    
                    self.child_conn.send(env_info)
                
                else:
                    # Unknown command
                    print(f"[Worker {self.env_idx}] Unknown command: {cmd}")
        
        except Exception as e:
            print(f"[Worker {self.env_idx}] Error in worker: {e}")
            import traceback
            traceback.print_exc()
            self.child_conn.close()


def create_parallel_envs(env_id, num_envs, log_dir=None, fixed_goal=None, fixed_start=None, seed=None, 
                        continuing_task=True, reset_target=False, use_dense_reward=False, 
                        max_episode_steps=500, maze_map=None, disable_rendering=True):
    """
    Create multiple environments in parallel
    
    Args:
        env_id (str): Environment ID
        num_envs (int): Number of parallel environments
        log_dir (str): Directory for logs
        fixed_goal (numpy.ndarray): Fixed goal for all environments
        fixed_start (numpy.ndarray): Fixed start position for all environments
        seed (int): Random seed
        continuing_task (bool): Whether the task is continuing
        reset_target (bool): Whether to reset target when reached
        use_dense_reward (bool): Whether to use dense rewards
        max_episode_steps (int): Maximum steps per episode
        maze_map (list): Optional custom maze map
        disable_rendering (bool): Whether to disable rendering for all environments
        
    Returns:
        tuple: (parent_conns, workers) - Pipes and worker processes
    """
    parent_conns = []
    workers = []
    
    # Create workers
    for idx in range(num_envs):
        # Create communication pipe
        parent_conn, child_conn = Pipe()
        
        # Create and start worker
        worker = WorkerProcess(
            env_id=env_id,
            is_render=False if disable_rendering else (idx == 0),  # Only render first environment if not disabled
            env_idx=idx,
            child_conn=child_conn,
            seed=seed,
            max_episode_steps=max_episode_steps,
            fixed_goal=fixed_goal,
            fixed_start=fixed_start,  # Pass fixed start position
            log_dir=log_dir,
            continuing_task=continuing_task,
            reset_target=reset_target,
            use_dense_reward=use_dense_reward,
            maze_map=maze_map
        )
        worker.start()
        
        parent_conns.append(parent_conn)
        workers.append(worker)
    
    # Get environment info from first worker
    parent_conns[0].send(('get_env_info', None))
    env_info = parent_conns[0].recv()
    
    print(f"Created {num_envs} environments")
    print(f"Observation dim: {env_info['obs_dim']}, Action dim: {env_info['action_dim']}")
    
    return parent_conns, workers, env_info