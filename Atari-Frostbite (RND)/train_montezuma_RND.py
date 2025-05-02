from agents_RND import *
from envs_RND import *
from utils_RND import *
from config_RND import *
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections import defaultdict

import numpy as np
import os
from datetime import datetime
import argparse


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    # Set the random seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Print the config hyperparameters
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config.get('TrainMethod')
    # Select the training environement
    env_id = default_config.get('EnvID')
    #  Select the env_type
    env_type = default_config.get('EnvType')

    if env_type == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        print("ERROR: This environment is not implemented yet!")
        raise NotImplementedError

    # Get the state size and action space
    input_size = env.observation_space.shape
    output_size = env.action_space.n  
    env.close()

    is_load_model = False
    is_render = False

    # Define the model path names
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models/{}-seed-{}.model'.format(env_id, args.seed))
    predictor_path = os.path.join(script_dir, 'models/{}-seed-{}.pred'.format(env_id, args.seed))
    target_path = os.path.join(script_dir, 'models/{}-seed-{}.target'.format(env_id, args.seed))

    # Get current datetime and hostname
    current_datetime = datetime.now().strftime("%m%d_%H-%M-%S")

    log_dir = os.path.join(script_dir, 'runs/{}_HEX_seed-{}'.format(current_datetime, args.seed))
    # Set the writer (Tensorboard)
    writer = SummaryWriter(log_dir=log_dir)

    # Get the config hyperparameters
    use_cuda = default_config.getboolean('UseGPU')
    
    # GAE hyperparameters
    use_gae = default_config.getboolean('UseGAE')
    lam = default_config.getfloat('Lambda')

    use_noisy_net = default_config.getboolean('UseNoisyNet')
    
    # Number of different instances of environments we're going to run in parallel
    num_worker = default_config.getint('NumEnv')

    num_step = default_config.getint('NumStep')

    # PPO epsilon (aka what will help us to define the cliprange )
    ppo_eps = default_config.getfloat('PPOEps')
    epoch = default_config.getint('Epoch')
    mini_batch = default_config.getint('MiniBatch')
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = default_config.getfloat('LearningRate')
    entropy_coef = default_config.getfloat('Entropy')
    
    # Extrinsic reward discount rate
    gamma = default_config.getfloat('Gamma')

    # Intrinsic reward discount rate
    int_gamma = default_config.getfloat('IntGamma')

    # Gradient normalization clip
    clip_grad_norm = default_config.getfloat('ClipGradNorm')

    # Extrinsic reward coefficient
    ext_coef = default_config.getfloat('ExtCoef')

    # Intrinsic reward coefficient
    int_coef = default_config.getfloat('IntCoef')

    # Use sticky action
    sticky_action = default_config.getboolean('StickyAction')
    action_prob = default_config.getfloat('ActionProb')
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = default_config.getint('ObsNormStep')
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    # Instantiate the agent
    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        seed=args.seed
    )

   # Loads models
    if is_load_model:
        if use_cuda:
            print("Loading PPO Saved Model using GPU")
            if torch.cuda.device_count() > 1:
                agent.model.module.load_state_dict(torch.load(model_path))
                agent.rnd.predictor.module.load_state_dict(torch.load(predictor_path))
                agent.rnd.target.module.load_state_dict(torch.load(target_path))
            else:
                agent.model.load_state_dict(torch.load(model_path))
                agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
                agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            print("Loading PPO Saved Model using CPU")
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))

    works = []
    parent_conns = []
    child_conns = []

    # Generate the different environements
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.seed(args.seed)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0
    update_return_sum = 0
    update_episode_count = 0
    visited_rooms_global = set()

    # normalize obs
    print('Start to initialize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')

    while sample_episode < 6000:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1
        
        # For calculating mean number of rooms visited per update
        visited_rooms_per_worker = defaultdict(set)

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn, work in zip(parent_conns, works):
                s, r, d, rd, lr = parent_conn.recv()

                # Get current room and add it to the set
                ram = unwrap(work.env).ale.getRAM()
                assert len(ram) == 128
                current_room = int(ram[3])
                visited_rooms_global.add(current_room)
                visited_rooms_per_worker[work.env_id].add(current_room)

                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1

                # Accumulate returns and episode count for averaging per update
                update_return_sum += sample_rall
                update_episode_count += 1

                # Calculate the mean episodic return
                mean_episodic_return = sample_rall / sample_step if sample_step > 0 else 0

                # Log mean episodic return
                writer.add_scalar('data/mean_episodic_return', mean_episodic_return, sample_episode)

                # Log cumulative rewards per episode (as before)
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)

                # Reset episode tracking variables
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training
        actor_loss, critic_loss, entropy, forward_loss, total_loss = agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)
        
        # Step 6. Logging Stuff
        writer.add_scalar('data/actor_loss', actor_loss, global_update)
        writer.add_scalar('data/critic_loss', critic_loss, global_update)
        writer.add_scalar('data/entropy', entropy, global_update)
        writer.add_scalar('data/forward_loss', forward_loss, global_update)
        writer.add_scalar('data/total_loss', total_loss, global_update)
        if update_episode_count > 0:
            # Log mean episodic return per parameter update
            mean_update_return = update_return_sum / update_episode_count
            writer.add_scalar('data/mean_epi_return_per_update', mean_update_return, global_update)

            # Calculate the mean number of rooms visited per update
            total_number_of_rooms = sum(len(s) for s in visited_rooms_per_worker.values())
            mean_rooms_visited = total_number_of_rooms / num_worker
            writer.add_scalar('data/mean_rooms_per_update', mean_rooms_visited, global_update)

            # Reset update tracking variables
            update_return_sum = 0
            update_episode_count = 0

        if global_step % (num_worker * num_step * 100) == 0:
            print("Num Step: ", num_step)
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)

    file_dir = os.path.join(script_dir, 'visited_rooms/{}visited_rooms{}.txt'.format(current_datetime, args.seed))
    with open(file_dir, "w") as file:
        for item in visited_rooms_global:
            file.write(f"{item}\n")

if __name__ == '__main__':
    main()
