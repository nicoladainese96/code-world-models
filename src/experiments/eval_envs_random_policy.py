import functools
import gymnasium as gym
import numpy as np
import os
import json
import time
import pickle
import sys
import copy
import fire
import torch 
from importlib import reload
import autopep8
import traceback

import mujoco_py

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.environments.planning import test_MCTS, test_MCTS_CWM, test_CEM, test_CEM_CWM
from src.environments import Simulator, GymSimulator, GymSimulatorValidActions, CWMSimulatorMCTS, CWMSimulatorCEM

import warnings
warnings.filterwarnings('ignore')

env_ids_and_planners = {
    'Blackjack-v1':'mcts', # 0
    'CliffWalking-v0':'mcts', # 1 
    'Taxi-v3':'mcts', # 2
    'CartPole-v1':'mcts', # 3 
    'MountainCar-v0':'mcts', # 4
    'Acrobot-v1':'mcts', # 5
    'Pendulum-v1':'cem', # 6
    'Reacher-v4':'cem', # 7
    'Pusher-v4':'cem', # 8
    'InvertedPendulum-v4':'cem', # 9
    'InvertedDoublePendulum-v4':'cem', # 10
    'HalfCheetah-v4':'cem', # 11
    'Hopper-v4':'cem', # 12
    'Swimmer-v4':'cem', # 13
    'Walker2d-v4':'cem', # 14
    'Ant-v4':'cem', # 15
    'Humanoid-v4':'cem', # 16 
    'HumanoidStandup-v4':'cem' # 17
}

def ensure_float(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value)
        else:
            raise ValueError("R is a numpy array with more than one entry.")
    return float(value)


def ran_random_episodes(env_id, n_episodes, verbose=False):
    real_env = gym.make(env_id)
    if hasattr(real_env, '_max_episode_steps'):
        max_episode_steps = real_env._max_episode_steps
    else:
        max_episode_steps = 100 # if not defined
    
    scores = []
    for i in range(n_episodes):
        frame, _ = real_env.reset()
        done = False
        R = 0
        t = 0
        while not done:
            t += 1
            action = real_env.action_space.sample()
            if verbose:
                print("Action: ", action)
            frame, r, done, truncated, extra_info = real_env.step(action)
            if verbose:
                print("Frame: ", frame)
                print("Reward: ", r)
            R += ensure_float(r)
            
            if t == max_episode_steps:
                break
        scores.append(R)
        
        print(f"Episode {i} completed with {t} steps (max is {max_episode_steps}) and a total reward of {R}")
    return scores

def main(
    save_dir,
    n_episodes=10,
):
    """
    save_dir: (str)
        Relative path from PROJECT_ROOT for saving the results. Results will be saved as a results.json, 
        cem_config.json and mcts_config.json .
        results.json is a dict with env_id:list_of_returns
    """
    start_time = time.time()
    
    NUM_ENVS = 18
    env_indexes = [i for i in range(NUM_ENVS)]
        
    list_of_env_ids = list(env_ids_and_planners.keys())
    env_ids = [list_of_env_ids[env_idx] for env_idx in env_indexes]

    results = {}
    for env_id in env_ids:
        scores = ran_random_episodes(env_id, n_episodes)
        results[env_id] = scores
    
    if not os.path.exists(os.path.join(PROJECT_ROOT, save_dir)):
        os.makedirs(os.path.join(PROJECT_ROOT, save_dir), exist_ok=True)
                                       
    # Write to save_dir results.json
    with open(os.path.join(PROJECT_ROOT, save_dir, 'results.json'), "w") as f:
        json.dump(results, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Everything done. Time elapsed: {elapsed_time//60} minutes.")
    
if __name__=='__main__':
    fire.Fire(main)