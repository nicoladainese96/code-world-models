import os
import sys
import json
import time
import fire
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.apps.generate_solutions import generate_solutions

def main(
    idx,
    total_tasks,
    model,
    hf_cache=None,
    mock=False,
    dataset_path="data/APPS/test",
    test_loc="data/APPS/test.json",
    save_dir="results/apps/default_exp",
    start_idx=3000,
    end_idx=4000,
    strategy="zero-shot-CoT",
    budget=1,
    **kwargs):
    
    # Divide range of problems in total_tasks equal splits
    range_size = (end_idx - start_idx) // total_tasks
    
    # Select the i-th split and define (start,end) accordingly
    start = start_idx + idx * range_size
    end = start + range_size if idx < total_tasks - 1 else end_idx
    
    # Call generate_solutions script
    generate_solutions(
        model,
        hf_cache,
        start,
        end,
        mock,
        dataset_path,
        test_loc,
        save_dir,
        strategy,
        budget,
        **kwargs
    )
    
if __name__ == "__main__":
    fire.Fire(main)