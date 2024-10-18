import os
import sys
import fire
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.cwm_benchmark.generate_env import generate_environment

def main(
    idx,
    model,
    env=None,
    hf_cache=None,
    budget=1,
    code_block_length=2,
    total_code_length=1000,
    eps=1,
    ucb_c=0.1,
    discount=1.0,
    v_g_new_init=0.5,
    g_counts_init=2,
    v_f_new_init=0.5,
    f_counts_init=2,
    v_i_new_init=0.55,
    i_counts_init=2, 
    fixes=3,
    log_file=False,
    log_level="INFO",
    save_dir="results/cwm/default_exp",
    allow_generate=True,
    allow_improve=True,
    **kwargs):
    
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
    
    # Form mcts params
    mcts_params = dict(
        node_length=code_block_length,
        ucb_c=ucb_c,
        discount=discount,
        max_actions=int(total_code_length/code_block_length),
        eps=eps,
        v_g_new_init=v_g_new_init,
        g_counts_init=g_counts_init,
        v_f_new_init=v_f_new_init,
        f_counts_init=f_counts_init,
        v_i_new_init=v_i_new_init,
        i_counts_init=i_counts_init, 
        max_fix_chain_length=fixes,
        allow_generate=allow_generate,
        allow_improve=allow_improve,
    )
    
    log_params = {
        'log_file': log_file,
        'log_level': log_level,
    }

    # Call generate_solutions script
    generate_environment(
        model=model,
        env=environment,
        hf_cache=hf_cache,
        budget=budget,
        save_dir=save_dir,
        mcts_params=mcts_params,
        log_params=log_params,
        **kwargs
    )
    
if __name__ == "__main__":
    fire.Fire(main)