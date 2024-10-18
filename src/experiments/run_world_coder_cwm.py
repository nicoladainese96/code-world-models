import os
import sys
import fire
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.cwm_benchmark.generate_solutions_world_coder import generate_solutions

def main(
    idx,
    model,
    env=None,
    hf_cache=None,
    budget=1,
    log_file=False,
    log_level="INFO",
    save_dir="results/cwm/default_exp",
    **kwargs):

    log_params = {
        'log_file': log_file,
        'log_level': log_level,
    }
    
    # Find the list of all environments
    envs = os.listdir(os.path.join(PROJECT_ROOT, 'data', 'replay_buffers', 'gymnasium_envs'))
    envs.append('rtfm')
    # envs = [
    #     'Pendulum-v1',
    #     'Reacher-v4',
    #     'Pusher-v4',
    #     'InvertedPendulum-v4',
    #     'InvertedDoublePendulum-v4',
    #     'HalfCheetah-v4',
    #     'Hopper-v4',
    #     'Swimmer-v4',
    #     'Walker2d-v4',
    #     'Ant-v4',
    #     'Humanoid-v4',
    #     'HumanoidStandup-v4'
    # ]
    assert idx < len(envs), f"idx {idx} is larger than the number of environments: {len(envs)}"
    environment = envs[idx]
    
    if env is not None and env in envs:
        environment = env
    print(f"Running environment: {environment}")

    # Call generate_solutions script
    generate_solutions(
        model,
        environment,
        hf_cache,
        budget,
        save_dir,
        log_params,
        **kwargs
    )
    
if __name__ == "__main__":
    fire.Fire(main)