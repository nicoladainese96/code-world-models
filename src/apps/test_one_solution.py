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
import multiprocessing
# example of checking the start method
from multiprocessing import get_start_method, set_start_method
import signal

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import src.apps.testing_util as test_util
from src.apps.merge_codes import combine_codes

EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10

def print_results(results: Dict, args:argparse.Namespace=None):
    """
    Given the results evaluated against the testcases we output some statistics.

    results is a dictionary with keys being the problem index and values being a list of lists of results for each solution.
    """
    strict_acc_pass_at_k_list = []
    avg_acc_pass_at_k_list = []
    k = None
    for index in results:
        # get num solutions per problem (assuming and enforcing that all problems have the same number of solutions)
        num_solutions = len(results[index])
        if k is None:
            k = num_solutions
        else:
            assert k == num_solutions, "Number of solutions per problem should be the same for every problem!"
        prob_strict_acc = []
        prob_avg_acc = []

        for i in range(num_solutions):
            problem_results = np.asarray(results[index][i]) # numpy array with num_tests[index][i] elements (varies among problems and solutions)
            sol_strict_acc = np.all(problem_results > 0)
            sol_avg_acc = np.mean(problem_results > 0)

            prob_strict_acc.append(sol_strict_acc)
            prob_avg_acc.append(sol_avg_acc)

        # Now pass@k metrics are the average across problems of the best-out-of-k solutions, so we need to take the max over all solutions
        strict_acc_pass_at_k_list.append(max(prob_strict_acc))
        avg_acc_pass_at_k_list.append(max(prob_avg_acc))    

 
    print(f"Test Case Average (pass@{k}) = {np.mean(avg_acc_pass_at_k_list)}")
    print(f"Strict Accuracy (pass@{k}) = {np.mean(strict_acc_pass_at_k_list)}")

def check_correctness(prob_path, generation, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug, result):
        result.append(test_util.run_test(prob_path=prob_path, test=generation, debug=debug))

    # The multiprocessing might cause further problems
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0]

# stuff for setting up signal timer
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)


def check_correctness_with_errors(prob_path, generation, timeout, debug=True, mp=True):
    return check_correctness_with_errors_queue(prob_path, generation, timeout, debug=debug, mp=mp)
    # return check_correctness_with_errors_manager(prob_path, generation, timeout, debug=debug, mp=mp)

def check_correctness_with_errors_manager(prob_path, generation, timeout, debug=True, mp=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug, result, errors):
        res, err = test_util.run_test_with_errors(prob_path=prob_path, test=generation, debug=debug, timeout=4)
        result.append(res)
        errors.append(err)

    if mp:
        if debug:
            # get the start method
            method = get_start_method()
            # report the start method
            print("method:", method)
            print("Init manager")
        with multiprocessing.Manager() as manager:
            #manager = multiprocessing.Manager()
            if debug:
                print("manager.list()")
            result = manager.list()
            if debug:
                print("manager.list()")
            errors = manager.list()
            if debug:
                print("Defining multiprocessing.Process")
            p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, result, errors))
            if debug:
                print("Starting process")
            p.start()
            if debug:
                print("Joining process")
            p.join(timeout=timeout + 1)
            if p.is_alive():
                p.kill()

    else:
        signal.alarm(timeout)
        try:
            if debug:
                print("Eval without multi-processing")
            result = []
            errors = []
            res, err = test_util.run_test_with_errors(prob_path=prob_path, test=generation, debug=debug, timeout=1)
            result.append(res)
            errors.append(err)
            signal.alarm(0)
        except Exception as e:
            result = False
            signal.alarm(0)
            
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        errors = [["Timeout exception"]* avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0], errors[0]

def check_correctness_with_errors_queue(prob_path, generation, timeout, debug=True, mp=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug, results_and_errors_queue):
        res, err = test_util.run_test_with_errors(prob_path=prob_path, test=generation, debug=debug, timeout=4)
        results_and_errors_queue.put((res, err))

    if mp:
        result = []
        errors = []
        if debug:
            print("Defining multiprocessing.Queue")
        results_and_errors_queue = multiprocessing.Queue()
        if debug:
            print("Defining multiprocessing.Process")
        p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, results_and_errors_queue))
        if debug:
            print("Starting process")
        p.start()
        if debug:
            print("Joining process")
        p.join(timeout=timeout + 1)

        if debug:
            print("Killing process")
        if p.is_alive():
            p.kill()

        if debug:
            print("Emptying queue")
        while not results_and_errors_queue.empty():
            res, err = results_and_errors_queue.get()
            result.append(res)
            errors.append(err)

        

    else:
        signal.alarm(timeout)
        try:
            if debug:
                print("Eval without multi-processing")
            result = []
            errors = []
            res, err = test_util.run_test_with_errors(prob_path=prob_path, test=generation, debug=debug, timeout=1)
            result.append(res)
            errors.append(err)
            signal.alarm(0)
        except Exception as e:
            result = False
            signal.alarm(0)
            
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        errors = [["Timeout exception"]* avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0], errors[0]

def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))

    solutions = {}
    results = {}
    # Resolve path name for codes_loc and results_loc
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if os.path.exists(codes_loc):
        results_loc = os.path.join(args.save, f"all_results.json") 
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
         
    print(codes_loc, results_loc)

    # Load solutions (solutions)
    with open(codes_loc, "r") as f: 
        solutions = json.load(f)

    # Here we load the paths to the problems in input and their tests
    if args.index:
        # Load single problem if arg.index is specified
        problems = [problems[args.index]]
    else:
        # Load [args.start:args.end] range of problems
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    print("Amount of problems to be evaluated: ",len(problems))
    print("Amount of solutions found: ", len(solutions))
    
    # Main eval loop
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            output_str = solutions[str(index+args.start)] # this is a list of strings (see below, where we loop over them)
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
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
 
        results[index+args.start+args.index] = res
        
        # Save results after each problem to avoid losing all results if the program crashes; 
        # File is either all_results.json or {args.start}-{args.end}_results.json
        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results


def main(args):
    # probably useless
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # combine the codes from multiple files
    combine_codes(args.save)
    
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    
    if args.print_results:
        # Eval already done, just read results and print them
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.json") 
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
        with open(results_loc, "r") as f: 
            results = json.load(f)
    else:
        # Eval to do, then save results and print them
        results = eval_and_save_problems(args)

    print_results(results, args) 


if __name__ == "__main__":
    import doctest
    doctest.testmod() # runs: print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="data/APPS/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default=".", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results/apps", help="Where the evaluated data is loaded from and results saved to.")
 
    args = parser.parse_args()

    main(args)
