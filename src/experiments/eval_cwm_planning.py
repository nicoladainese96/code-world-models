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

def run_real_eval_mcts(env_id, n_episodes, max_actions, num_simulations, T, save_dir):
    mcts_params = {
        'ucb_c':1.0,
        'discount':0.99,
        'max_actions':max_actions
    }
    
    real_env = gym.make(env_id)
    valid_actions = np.arange(real_env.action_space.n)
    
    # Define Gym Simulator compatible with MCTS
    simulator = GymSimulatorValidActions(real_env, valid_actions)
    
    # currenty hangs on Taxi-v3 ... maybe it's very heavy to deepcopy it?
    scores = test_MCTS(real_env, simulator, mcts_params, T, num_simulations, n_episodes=n_episodes)
        
    print(f"Average score for {env_id}: {np.mean(scores)}")
    return scores

def run_cwm_eval_mcts(env_id, n_episodes, max_actions, num_simulations, T, save_dir, experiment_name):
    mcts_params = {
        'ucb_c':1.0,
        'discount':0.99,
        'max_actions':max_actions
    }
    
    code_path = os.path.join(PROJECT_ROOT, 'results', 'gymnasium_envs', experiment_name, env_id, 'best_code_rollout.py')
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            full_program = f.read()

        # Temporarily write code as importable gen_code_world_model module
        full_program = autopep8.fix_code(full_program)
        with open('gen_code_world_model.py', 'w') as f:
            f.write(full_program)

        try:
            # Import generated code
            import gen_code_world_model

            # Force reloading of the module
            gen_code_world_model = reload(gen_code_world_model)

            # Generate an instance of the Environment class
            code_env = gen_code_world_model.Environment()

            # Define real env
            real_env = gym.make(env_id)
            valid_actions = np.arange(real_env.action_space.n)
            
            # Define CWM Simulator
            cwm_simulator = CWMSimulatorMCTS(code_env, valid_actions)

            scores = test_MCTS_CWM(real_env, cwm_simulator, mcts_params, T, num_simulations, n_episodes=n_episodes)
        except Exception as e:
            print(f"Evaluation failed:")
            print(traceback.format_exc())
            scores = [None]*n_episodes
    else:
        print("No Code World Model found for the current environment.")
        scores = [None]*n_episodes
    if scores[0] is not None:
        print(f"Average score for {env_id}: {np.mean(scores)}")
    return scores

def run_real_eval_cem(env_id, n_episodes, T, I, N, K, save_dir):
    cem_params = {
        'T':T,
        'I':I,
        'N':N,
        'K':K
    }
    
    # Define real env
    real_env = gym.make(env_id)
    
    # currenty hangs on Taxi-v3 ... maybe it's very heavy to deepcopy it?
    scores = test_CEM(real_env, cem_params, n_episodes=n_episodes)
        
    print(f"Average score for {env_id}: {np.mean(scores)}")
    return scores

def run_cwm_eval_cem(env_id, n_episodes, T, I, N, K, experiment_name, save_dir):
    cem_params = {
        'T':T,
        'I':I,
        'N':N,
        'K':K
    }
    code_path = os.path.join(PROJECT_ROOT, 'results', 'gymnasium_envs', experiment_name, env_id, 'best_code_rollout.py')
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            full_program = f.read()

        # Temporarily write code as importable gen_code_world_model module
        full_program = autopep8.fix_code(full_program)
        with open('gen_code_world_model.py', 'w') as f:
            f.write(full_program)

        try:
            # Import generated code
            import gen_code_world_model

            # Force reloading of the module
            gen_code_world_model = reload(gen_code_world_model)

            # Generate an instance of the Environment class
            code_env = gen_code_world_model.Environment()
            
            # Define real env
            real_env = gym.make(env_id)

            scores = test_CEM_CWM(real_env, code_env, cem_params, n_episodes=n_episodes)
        except Exception as e:
            print(f"Evaluation failed:")
            print(traceback.format_exc())
            scores = [None]*n_episodes
    else:
        print("No Code World Model found for the current environment.")
        scores = [None]*n_episodes
    if scores[0] is not None:
        print(f"Average score for {env_id}: {np.mean(scores)}")
    return scores

def main(
    save_dir,
    eval_real=False, 
    experiment_name=None,
    env_indexes=[], # default to eval on all
    n_episodes=2,
    T_cem=100,
    I_cem=20,
    N_cem=1000,
    K_cem=100,
    max_actions_mcts=100,
    num_simulations_mcts=25,
    T_mcts=0.01
):
    """
    save_dir: (str)
        Relative path from PROJECT_ROOT for saving the results. Results will be saved as a results.json, 
        cem_config.json and mcts_config.json .
        results.json is a dict with env_id:list_of_returns
    eval_real: (bool)
        True -> benchmark real envs, False -> benchmark code envs
    experiment_name: (str) - if eval_real is False
        Should be a folder under results/gymnasium_envs/ containing inside a folder for each
        env_id to be tested. Inside each folder, there should be a file called best_code_rollout.py
    env_indexes: (list of int)
        List of all the environments to be tested according to the convention defined in env_ids_and_planners
        above. Empty list defaults to all environments.
    T_cem: (int)
        Time horizon used in planning with Cross-Entropy Method (CEM)
    I_cem: (int)
        Amount of iterations done in CEM
    N_cem: (int)
        Amount of samples used in CEM
    K_cem: (int)
        Amount of elites used in CEM
    max_actions_mcts: (int)
        Number of maximum actions used in a rollout inside MCTS
    num_simulations_mcts: (int)
        Number of simulations used in MCTS to pick the next action to execute in the real environment
    T_mcts: (float in [0,1])
        Temperature used in the softmax applied to the root actions' Q-values to generate the policy from 
        which the next action (to be executed in the real env) is sampled
    """
    start_time = time.time()
    
    NUM_ENVS = 18
    
    if len(env_indexes)==0:
        env_indexes = [i for i in range(NUM_ENVS)]
        
    if not eval_real:
        assert experiment_name is not None, "Must specify experiment_name!"
        
    list_of_env_ids = list(env_ids_and_planners.keys())
    env_ids = [list_of_env_ids[env_idx] for env_idx in env_indexes]
    
    mcts_config = {
        'max_actions':max_actions_mcts,
        'num_simulations':num_simulations_mcts,
        'T':T_mcts
    }
    
    cem_config = {
        'T':T_cem,
        'I':I_cem,
        'N':N_cem,
        'K':K_cem
    }
    
    results = {}
    for env_id in env_ids:
        planning_method = env_ids_and_planners[env_id]
        if eval_real:
            if planning_method=='mcts':
                scores = run_real_eval_mcts(env_id=env_id, n_episodes=n_episodes, save_dir=save_dir, **mcts_config)
            else:
                scores = run_real_eval_cem(env_id=env_id, n_episodes=n_episodes, save_dir=save_dir, **cem_config)
        else:
            if planning_method=='mcts':
                scores = run_cwm_eval_mcts(env_id=env_id, n_episodes=n_episodes, save_dir=save_dir, experiment_name=experiment_name, **mcts_config)
            else:
                scores = run_cwm_eval_cem(env_id=env_id, n_episodes=n_episodes, save_dir=save_dir, experiment_name=experiment_name, **cem_config)
            
        results[env_id] = scores
    
    if not os.path.exists(os.path.join(PROJECT_ROOT, save_dir)):
        os.makedirs(os.path.join(PROJECT_ROOT, save_dir), exist_ok=True)
                                       
    # Write to save_dir results.json
    with open(os.path.join(PROJECT_ROOT, save_dir, 'results.json'), "w") as f:
        json.dump(results, f)
        
    # Write to save_dir mcts_config.json
    with open(os.path.join(PROJECT_ROOT, save_dir, 'mcts_config.json'), "w") as f:
        json.dump(mcts_config, f)
        
    # Write to save_dir cem_config.json
    with open(os.path.join(PROJECT_ROOT, save_dir, 'cem_config.json'), "w") as f:
        json.dump(cem_config, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Everything done. Time elapsed: {elapsed_time//60} minutes.")
    
if __name__=='__main__':
    fire.Fire(main)