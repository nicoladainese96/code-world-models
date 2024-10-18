import os
import sys
import torch
import copy
import numpy as np

# Import src code from parent directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath('')))
sys.path.append(PROJECT_ROOT)

from .replay_buffer import ReplayBuffer

def print_board(board):
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            print(str(board[i, j, 0]).ljust(20), end=' ')
        print()

action_dict = {0: "Stay", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}

def print_trajectory(trajectory):
    states, actions, next_states, rewards, dones, extra_infos = trajectory
    for i in range(len(states)):
        print(f"Step {i}")
        print_board(states[i][0])
        print("Inventory: ", states[i][1])
        print()
        print("Action: ", action_dict[actions[i]], actions[i])
        print("Reward: ", rewards[i])
        print("Done: ", dones[i])
        print()

def collect_trajectory(env, max_steps=100, gymnasium_interface=False):
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    extra_infos = []
    if gymnasium_interface:
        obs, _ = env.reset()
        valid_actions = env.action_space
    else:
        obs, valid_actions, _ = env.reset()

    for t in range(max_steps):
        # print(f"Step {t}")
        if gymnasium_interface:
            action = valid_actions.sample()
        else:
            action = np.random.choice(valid_actions)
        states.append(copy.deepcopy(obs))
        # print_board(obs[0])
        # print("Inventory: ", obs[1])
        # print(obs)
        # print("Action: ", action_dict[action], action)
        if gymnasium_interface:
            next_obs, reward, terminated, truncated, extra_info = env.step(action)
            done = terminated or truncated
        else:
            next_obs, valid_actions, reward, done, extra_info = env.step(action)
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


def fill_buffer(env, capacity=50000, file_path=PROJECT_ROOT, buffer_name="buffer", force_new=False, max_attempts=1000,
                traj_filter=None, gymnasium_interface=False, max_episode_length=100, verbose=False):
    """Usage:
        fill_buffer(env, capacity=n, max_episode_length=l)
            to collect exactly n transitions with maximum episode length l
        fill_buffer(env, capacity=None, max_attempts=n_max, traj_filter=trajectory_filter_fn, max_episode_length=l)
            to collect trajectory_filter_fn.n_to_collect trajectories according to a filter function with maximum episode length l.
    """
    buffer_file = os.path.join(file_path, f"{buffer_name}.pkl")
    device = torch.device("cpu")

    if os.path.exists(buffer_file) and not force_new:
        print(f"Buffer {buffer_name} already exists, loading it.")
        buffer = ReplayBuffer(
            capacity=capacity,
            device=device,
            file_path=file_path,
            buffer_name=buffer_name
        )
        buffer.load()
        return buffer
    
    if traj_filter is not None:
        capacity = traj_filter.n_to_collect * max_episode_length
        print('Setting capacity', capacity)
    buffer = ReplayBuffer(
        capacity=capacity,
        device=device,
        file_path=file_path,
        buffer_name=buffer_name
    )
    n_attempts = 0
    traj_lengths = {}

    while not buffer.full:
        try:
            trajectory = collect_trajectory(env, max_episode_length, gymnasium_interface=gymnasium_interface)
            # Collect every trajectory until buffer is full.
            if traj_filter is None:
                if len(trajectory[0]) > buffer.capacity - buffer.idx:
                    break
                buffer.add_trajectory(trajectory)
            else:
                if traj_filter.check_trajectory(trajectory[3]):
                    buffer.add_trajectory(trajectory)
                    if verbose:
                        print(np.array(trajectory[3]))
        except ValueError as e:
            print("No legal action found, skipping trajectory")
            print(e)
            pass
        n_attempts += 1
        traj_len = len(trajectory[3])
        if traj_len in traj_lengths:
            traj_lengths[traj_len] += 1
        else:
            traj_lengths[traj_len] = 1
        if traj_filter is not None:
            if traj_filter.is_finished:
                break
            elif max_attempts is not None and n_attempts >= max_attempts:
                break

    if verbose:
        print('Trajectory lengths:',
              sorted(traj_lengths.items(), key=lambda x: x[1], reverse=True))
    buffer.save()
    return buffer

def fill_buffer_binary_outcome(env, capacity=50000, file_path=PROJECT_ROOT, buffer_name="buffer", force_new=False,
                               n_successes=None, n_failures=None, max_attempts=1000, success_threshold=0, success_criteria=None,
                               gymnasium_interface=False, verbose=False):
    """Usage:
        fill_buffer(env, capacity=n)
            to collect exactly n transitions
        fill_buffer(env, capacity=None, n_successes=s, n_failures=f, max_attempts=n_max, success_threshold=r)
            to collect s success and f failure episodes (with maximum n_max episode attempts), where r is the minimum
            reward considered a success.
    """
    buffer_file = os.path.join(file_path, f"{buffer_name}.pkl")
    device = torch.device("cpu")

    if os.path.exists(buffer_file) and not force_new:
        print(f"Buffer {buffer_name} already exists, loading it.")
        buffer = ReplayBuffer(
            capacity=capacity,
            device=device,
            file_path=file_path,
            buffer_name=buffer_name
        )
        buffer.load()
        if n_successes is None or n_failures is None:
            return buffer
        return buffer, True
 
    assert (n_successes is None and n_failures is None) or (
        n_successes is not None and n_failures is not None), "Either both n_successes and n_failures must be None or both must be specified."
    max_episode_length = env._max_episode_steps if hasattr(env, '_max_episode_steps') else 100
    if n_failures is not None and n_successes is not None:
        capacity = (n_successes + n_failures) * max_episode_length
        print('Setting capacity', capacity)
    buffer = ReplayBuffer(
        capacity=capacity,
        device=device,
        file_path=file_path,
        buffer_name=buffer_name
    )
    n_failures_collected = 0
    n_successes_collected = 0
    n_attempts = 0

    while not buffer.full:
        try:
            trajectory = collect_trajectory(env, gymnasium_interface=gymnasium_interface)
            # Collect every trajectory until buffer is full.
            if n_successes is None and n_failures is None:
                if len(trajectory[0]) > buffer.capacity - buffer.idx:
                    break
                buffer.add_trajectory(trajectory)
            else:
                if success_criteria is None:
                    is_success = np.any(np.array(trajectory[3]) >= success_threshold)
                else:
                    is_success = success_criteria(trajectory)
                if is_success:
                    if n_successes is not None and n_successes_collected < n_successes:
                        assert len(trajectory[0]) <= buffer.capacity - buffer.idx, f'Not enough space in buffer for success trajectory (length {len(trajectory[0])})'
                        buffer.add_trajectory(trajectory)
                        n_successes_collected += 1
                        print('Collected success', n_successes_collected)
                        if verbose:
                            print(np.array(trajectory[3]))
                else:
                    if n_failures is not None and n_failures_collected < n_failures:
                        assert len(trajectory[0]) <= buffer.capacity - buffer.idx, f'Not enough space in buffer for failure trajectory (length {len(trajectory[0])}'
                        buffer.add_trajectory(trajectory)
                        n_failures_collected += 1
                        print('Collected failure', n_failures_collected)
                        if verbose:
                            print(np.array(trajectory[3]))
            #print(f"Buffer capacity: {buffer.idx}/{buffer.capacity}")
        except ValueError as e:
            print("No legal action found, skipping trajectory")
            print(e)
            pass
        if n_successes is not None and n_failures is not None:
            if n_successes_collected >= n_successes and n_failures_collected >= n_failures:
                print(f'Finished in {max_attempts} attempts. '
                      f'Collected {n_successes_collected} successes and {n_failures_collected} failures.')
                break

            n_attempts += 1
            if max_attempts is not None and n_attempts >= max_attempts:
                print(f'Giving up after {max_attempts} attempts. '
                      f'Collected {n_successes_collected} successes and {n_failures_collected} failures.')
                break

    buffer.save()
    if n_successes is None or n_failures is None:
        return buffer
    finished = n_successes_collected >= n_successes and n_failures_collected >= n_failures
    return buffer, finished
