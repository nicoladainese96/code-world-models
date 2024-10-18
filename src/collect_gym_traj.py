import functools
import gymnasium as gym
import numpy as np
import os
import pickle
import sys
import copy
import torch 
import time

import mujoco_py

PROJECT_ROOT = os.path.abspath('')
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.cem import cross_entropy_method
from src.environments import GymSimulator
    #'MountainCarContinuous-v0', # requires exploration to overcome sparse rewards
    #'CarRacing-v2', # giving problems
    
list_of_env_ids = [
    'Pendulum-v1', 
    'Reacher-v4', 
    'Pusher-v4', 
    'InvertedPendulum-v4', 
    'InvertedDoublePendulum-v4', 
    'HalfCheetah-v4', 
    'Hopper-v4', 
    'Swimmer-v4', 
    'Walker2d-v4', 
    'Ant-v4', 
    'Humanoid-v4', 
    'HumanoidStandup-v4'
]

def collect_trajectory_cem(env, CEM_params, max_steps=100):
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    extra_infos = []
    
    
    
    obs, _ = env.reset()
    valid_actions = env.action_space
    action_plan = []
    
    
    for t in range(max_steps):
        # print(f"Step {t}")
        
        # If the action_plan is empty, synch the simulator to the real env and run CEM
        if len(action_plan) == 0:
            simulator = GymSimulator(env)
            # Generate action plan with CEM for next CEM_params['T'] steps
            action_plan = cross_entropy_method(simulator, **CEM_params)
            action_plan = [a for a in action_plan]
            
        # Follow the action plan until the list is empty
        action = action_plan.pop(0)
        
        states.append(copy.deepcopy(obs))
        # print_board(obs[0])
        # print("Inventory: ", obs[1])
        # print(obs)
        # print("Action: ", action_dict[action], action)

        next_obs, reward, terminated, truncated, extra_info = env.step(action)
        done = terminated or truncated

        # print("Reward: ", reward)
        # print("Done: ", done)
        # print()
        # print_board(next_obs[0])
        # print("Inventory: ", next_obs[1])
        # print(next_obs)
        obs = next_obs
        actions.append(action)
        rewards.append(reward)
        next_states.append(copy.deepcopy(next_obs))
        dones.append(done)
        extra_info['valid_actions'] = valid_actions
        extra_infos.append(copy.deepcopy(extra_info))
        if done:
            break

    # print("\n\nTrajectory")
    # print_trajectory((states, actions, next_states, rewards, dones, extra_infos))

    return states, actions, next_states, rewards, dones, extra_infos

from src.replay_buffer import ReplayBuffer

def fill_buffer_cem(env, CEM_params, file_path, buffer_name, force_new=True, max_episode_length=100, verbose=False, n_traj=10):
    
    def get_iterations_budget(n_traj):
        """
        10 -> [1,2,3,4,5,8,12,16,20,25]
        11 -> [1,2,3,4,5,6,8,12,16,20,25]
        12 -> [1,2,3,4,5,6,8,12,16,20,24,25]
        """
        return [i for i in range(1, n_traj//2 + n_traj%2 + 1)]+[4*i for i in range(2, n_traj//2 + 1)]+[25]

    buffer_file = os.path.join(file_path, f"{buffer_name}.pkl")
    device = torch.device("cpu")
    
    # Buffer already exists
    if os.path.exists(buffer_file) and not force_new:
        print(f"Buffer {buffer_name} already exists, loading it.")
        buffer = ReplayBuffer(
            capacity=n_traj*max_episode_length,
            device=device,
            file_path=file_path,
            buffer_name=buffer_name
        )
        buffer.load()
        return buffer
    
    # Buffer doesn't exists
    buffer = ReplayBuffer(
        capacity=n_traj*max_episode_length,
        device=device,
        file_path=file_path,
        buffer_name=buffer_name
    )
    
    iterations_budget = get_iterations_budget(n_traj)
    
    for i in range(n_traj):
        if (i == (n_traj-1)//2) or (i ==(n_traj-1)):
            print(f"Collecting trajectory {i+1} of {n_traj} with iteration budget {iterations_budget[i]}")
        CEM_params['I'] = iterations_budget[i]
        trajectory = collect_trajectory_cem(env, CEM_params, max_episode_length)
        buffer.add_trajectory(trajectory)
    
    buffer.save()
    return buffer

def main():
    T = 100
    I = 2
    N = 1000
    K = 100

    CEM_params = {
        'T':T,
        'I':I,
        'N':N,
        'K':K
    }

    incomplete_envs = []
    for i, env_id in enumerate(list_of_env_ids):
        print('-'*80)
        
        # Already finished
        if env_id in ['Acrobot-v1', 'CartPole-v1']:
            continue
        env = gym.make(env_id)
        buffer_name = 'train_buffer'
        file_path = os.path.join(PROJECT_ROOT, "data", "replay_buffers", "gymnasium_envs", env_id)
        # if os.path.exists(os.path.join(file_path, f'{buffer_name}.pkl')):
        #     print('Buffer exists for', env_id)
        #     continue
        if hasattr(env, '_max_episode_steps'):
            print(env_id, 'max episode length', env._max_episode_steps)
        else:
            print(env_id, 'has no max episode length')

        if hasattr(env, '_max_episode_steps'):
            max_steps = env._max_episode_steps + 1
            #print("Using environment's own step limit + 1:", max_steps)
        else:
            max_steps = 500 #max_episode_lengths[env_id]
            #print("Using default step limit (500)")
        
        max_steps = min(max_steps,100)
        print("Max steps per episode: ", max_steps)
        
        start = time.time()
        train_buffer = fill_buffer_cem(
            env,
            CEM_params=CEM_params,
            file_path=file_path,
            buffer_name=buffer_name,
            force_new=False,
            max_episode_length=max_steps,
            verbose=True,
            n_traj=10
        )
        end = time.time()
        t_elapsed = end-start
        print(f"Env {env_id} took {t_elapsed//60} min {t_elapsed%60} sec")
        n_to_go = len(list_of_env_ids)-i-1
        print(f'{n_to_go} environments to go might take approximately {t_elapsed//60*n_to_go} minutes more.')
        
if __name__=="__main__":
    main()