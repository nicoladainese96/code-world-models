import os
import sys
import json
import time
import fire
import signal
import numpy as np

import cProfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.apps.generate_solutions_gif_mcts import generate_solutions

def main(
    idx,
    total_tasks,
    model,
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
    test_loc="data/APPS/test.json",
    save_dir="results/apps/default_exp",
    start_idx=3000,
    end_idx=4000,
    **kwargs):
    
    # Divide range of problems in total_tasks equal splits
    range_size = (end_idx - start_idx) // total_tasks
    
    # Select the i-th split and define (start,end) accordingly
    start = start_idx + idx * range_size
    end = start + range_size if idx < total_tasks - 1 else end_idx
    
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
    )
    
    log_params = {
        'log_file': log_file,
        'log_level': log_level,
    }

    # Call generate_solutions script
    generate_solutions(
        model,
        hf_cache,
        budget,
        start,
        end,
        test_loc,
        save_dir,
        mcts_params,
        log_params,
        **kwargs
    )
    
if __name__ == "__main__":
    fire.Fire(main)