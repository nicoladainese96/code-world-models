import os
import sys
import fire
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.apps.generate_solutions_world_coder import generate_solutions

def main(
    idx,
    total_tasks,
    model,
    hf_cache=None,
    budget=1,
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
        log_params,
        **kwargs
    )
    
if __name__ == "__main__":
    fire.Fire(main)