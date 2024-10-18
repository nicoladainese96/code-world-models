# From https://github.com/hendrycks/apps/blob/main/eval/test_one_solution.py
"""
Run solutions from one problem.
"""
import sys
import argparse
import json
import numpy as np
import os
import pprint

from tqdm import tqdm
from typing import Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import src.apps.testing_util as test_util
from src.apps.test_one_solution import print_results, check_correctness
from src.apps.merge_codes import combine_codes

TIMEOUT = 10

def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f)) # full_paths to problems

    solutions = {}
    results = {}
    # Resolve path name for codes_loc and results_loc
    codes_loc = os.path.join(args.save, args.combined_name)
    results_loc = os.path.join(args.save, args.result_name) 
  
    print(codes_loc, results_loc)

    # Load solutions (solutions)
    with open(codes_loc, "r") as f: 
        solutions = json.load(f)

    print("Amount of solutions found: ", len(solutions))
    print("solutions.key()", solutions.keys())
    
    
    # Main eval loop
    for problem_id in solutions.keys():
        problem = problems[int(problem_id)]
        prob_path = os.path.join(args.root, problem)
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            output_str = solutions[problem_id] # this is a list of strings (see below, where we loop over them)
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem_id)
            continue

        res = [] # list of results for each solution -> list of lists
        for o_idx, o in enumerate(output_str): 
            if args.debug:
                print(f"\nTesting solution {o_idx}")
            curr_res = [-2] # default value is a compile error
            try:
                # Try checking correctness and overwriting curr_res
                curr_res = check_correctness(prob_path=prob_path, generation=o, timeout=TIMEOUT, debug=args.debug)
                # Fix the numpy array and boolean types
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
 
        results[problem_id] = res
        
        # Save results after each problem to avoid losing all results if the program crashes; 
        # File is either all_results.json or {args.start}-{args.end}_results.json
        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results

def combine_codes(root_dir, combined_name="partial_codes.json"):
    result_files = os.listdir(root_dir)
    tmp_codes = {}
   
    # load the results and combine them
    for r_file in result_files:
        path = os.path.join(root_dir, r_file)
        if "results.json" in path:
            continue
        elif "codes" in path and combined_name not in path:
            with open(path, "r") as f:
                results = json.load(f)
            for res in results:
                assert isinstance(results[res],list), "Problem in combining the results!"
                tmp_codes[res] = results[res]
            continue
    
    print("Amount of problems combined:", len(tmp_codes))
    with open(os.path.join(root_dir, combined_name), 'w') as f:
        json.dump(tmp_codes, f)


def main(args):
    # probably useless
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # combine the codes from multiple files
    combine_codes(args.save, args.combined_name)
    
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    
    results = eval_and_save_problems(args)
    print_results(results, args) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="data/APPS/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default=".", type=str, help="where the data is stored.")
    parser.add_argument("--combined_name", default="partial_codes.json", type=str, help="Name of the joint codes.")
    parser.add_argument("--result_name", default="partial_results.json", type=str, help="Name of the joint results.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results/apps", help="Where the evaluated data is loaded from and results saved to.")
 
    args = parser.parse_args()

    main(args)
