import os
import sys
import copy
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.mcts import StochasticMCTS, StochasticMCTS_CWM
from src.cem import cross_entropy_method, cross_entropy_method_cwm
from src.environments import Simulator, GymSimulator, GymSimulatorValidActions, CWMSimulatorMCTS, CWMSimulatorCEM

def ensure_float(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value)
        else:
            raise ValueError("R is a numpy array with more than one entry.")
    return float(value)


def test_MCTS(real_env, simulator, mcts_params, T, num_simulations, n_episodes=100, verbose=False):

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
            mcts = StochasticMCTS(frame, simulator, simulator.valid_actions, **mcts_params)
            root, extra_info = mcts.run(num_simulations)
            action, probs = root.softmax_Q(T)
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

def test_MCTS_CWM(real_env, cwm_simulator, mcts_params, T, num_simulations, n_episodes=100, verbose=False):

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
            mcts = StochasticMCTS_CWM(frame, cwm_simulator, cwm_simulator.valid_actions, **mcts_params)
            root, extra_info = mcts.run(num_simulations)
            action, probs = root.softmax_Q(T)
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

def test_CEM(real_env, cem_params, n_episodes=100, verbose=False):
    if hasattr(real_env, '_max_episode_steps'):
        max_episode_steps = real_env._max_episode_steps
    else:
        max_episode_steps = 100 # if not defined
    
    if cem_params['T'] > max_episode_steps:
        print(f"Changing time horizon of CEM from {cem_params['T']} to max_episode_steps ({max_episode_steps}) for this environment.")
        cem_params = copy.deepcopy(cem_params)
        cem_params['T'] = max_episode_steps
        
    scores = []
    for i in range(n_episodes):
        action_plan = []
        frame, _ = real_env.reset()
        done = False
        R = 0
        for t in range(max_episode_steps):

            # If the action_plan is empty, synch the simulator to the real env and run CEM
            if len(action_plan) == 0:
                print(f"Timestep {t} of {max_episode_steps} - reformulating action plan with CEM.")
                simulator = GymSimulator(real_env)
                # Generate action plan with CEM for next CEM_params['T'] steps
                action_plan = cross_entropy_method(simulator, **cem_params)
                print(f"Action plan information: type ({type(action_plan)}), length ({len(action_plan)})")
                if isinstance(action_plan, np.ndarray):
                    if action_plan.shape[0] != cem_params['T']:
                        assert action_plan.shape[0] == 1, f'No idea what is happening with the action_plan shape first dimension : {action_plan.shape}'
                        assert action_plan.shape[1] == cem_params['T'], f'No idea what is happening with the action_plan shape second dimension : {action_plan.shape}'
                        action_plan = action_plan[0]

                action_plan = [a for a in action_plan]
                print(f"Action plan information after post-processing: type ({type(action_plan)}), length ({len(action_plan)})")
                
            # Follow the action plan until the list is empty
            action = action_plan.pop(0)
            if verbose:
                print("Action: ", action)
            frame, r, done, truncated, extra_info = real_env.step(action)
            if verbose:
                print("Frame: ", frame)
                print("Reward: ", r)
            R += ensure_float(r)
            
            if done:
                break
        scores.append(R)
        print(f"Episode {i} completed with {t} steps (max is {max_episode_steps}) and a total reward of {R}")
    
    return scores

def test_CEM_CWM(real_env, code_env, cem_params, n_episodes=100, verbose=False):
    if hasattr(real_env, '_max_episode_steps'):
        max_episode_steps = real_env._max_episode_steps
    else:
        max_episode_steps = 100 # if not defined
    
    if cem_params['T'] > max_episode_steps:
        print(f"Changing time horizon of CEM from {cem_params['T']} to max_episode_steps ({max_episode_steps}) for this environment.")
        cem_params = copy.deepcopy(cem_params)
        cem_params['T'] = max_episode_steps
        
    scores = []
    for i in range(n_episodes):
        action_plan = []
        frame, _ = real_env.reset()
        done = False
        R = 0
        for t in range(max_episode_steps):

            # If the action_plan is empty, synch the simulator to the real env and run CEM
            if len(action_plan) == 0:
                print(f"Timestep {t} of {max_episode_steps} - reformulating action plan with CEM.")
                simulator = CWMSimulatorCEM(code_env)
                simulator.action_space = real_env.action_space
                # Generate action plan with CEM for next CEM_params['T'] steps
                action_plan = cross_entropy_method_cwm(simulator, frame, **cem_params)
                print(f"Action plan information: type ({type(action_plan)}), length ({len(action_plan)})")
                if isinstance(action_plan, np.ndarray):
                    if action_plan.shape[0] != cem_params['T']:
                        assert action_plan.shape[0] == 1, f'No idea what is happening with the action_plan shape first dimension : {action_plan.shape}'
                        assert action_plan.shape[1] == cem_params['T'], f'No idea what is happening with the action_plan shape second dimension : {action_plan.shape}'
                        action_plan = action_plan[0]

                action_plan = [a for a in action_plan] # Does this actually do something?
                print(f"Action plan information after post-processing: type ({type(action_plan)}), length ({len(action_plan)})")

            # Follow the action plan until the list is empty
            action = action_plan.pop(0)
            if verbose:
                print("Action: ", action)
            frame, r, done, truncated, extra_info = real_env.step(action)
            if verbose:
                print("Frame: ", frame)
                print("Reward: ", r)
            R += ensure_float(r)
            
            if done:
                break
        scores.append(R)
        print(f"Episode {i} completed with {t} steps (max is {max_episode_steps}) and a total reward of {R}")
    
    return scores