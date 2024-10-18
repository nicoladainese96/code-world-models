import os
import json

def load_problems(test_loc, start, end):
    with open(test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering
    
    if start > len(problems) or start < 0:
        print(f"start index {start} > number of problems {len(problems)}")
        return
    if end is None or end > len(problems):
        end = len(problems)
    return problems[start:end]

def get_problems_id_for_difficulty(files, dataset_path, difficulty='competition'):
    paths_for_difficulty = []
    for file in files:
        with open(os.path.join(dataset_path, file, 'metadata.json')) as f:
            metadata = json.load(f)
            if metadata['difficulty'] != difficulty:
                continue
            else:
                paths_for_difficulty.append(file)
    return paths_for_difficulty

def generate_prompt(test_case_path, prompt_path):
    _input = '<user>'
    with open(prompt_path, "r") as f:
        data = f.readlines()
        _input += "".join(data)

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nPlease read the inputs from the standard input (stdin) and print the outputs to the standard output (stdout).\n"
    else:
        _input += "\nUse Call-Based format.\n" # never used
    
    _input += "\nOutput your code solution with the following format:\n```python\n[your code]\n```</user>"
    return _input

def generate_CoT_prompt(test_case_path, prompt_path):
    _input = '<user>'
    with open(prompt_path, "r") as f:
        data = f.readlines()
        _input += "".join(data)

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nPlease read the inputs from the standard input (stdin) and print the outputs to the standard output (stdout).\n"
    else:
        _input += "\nUse Call-Based format.\n" # never used
    
    _input += "\nOutput your code solution with the following format:\n```python\n[your code]\n```</user>"
    _input += "<assistant>\nLet's think step by step.\n</assistant>"
    return _input

def generate_Plan_and_Solve_prompt(test_case_path, prompt_path):
    _input = '<user>'
    with open(prompt_path, "r") as f:
        data = f.readlines()
        _input += "".join(data)

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nPlease read the inputs from the standard input (stdin) and print the outputs to the standard output (stdout).\n"
    else:
        _input += "\nUse Call-Based format.\n" # never used
    
    _input += "\nOutput your code solution with the following format:\n```python\n[your code]\n```</user>"
    _input += "<assistant>\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.\n</assistant>"
    return _input

def backup_solutions(solutions, start, index, save_dir, logger):
    # Save solutions to file start-end_codes.json for parallel generation across jobs
    logger.info(f"Saving backup of partial solutions as {start}-{index+start}_codes.json")

    codes_loc = os.path.join(save_dir, f"{start}-{index+start}_codes.json")
    with open(codes_loc, "w") as f:
        json.dump(solutions, f)

    # Also remove previous backup
    if index > 0:
        logger.info(f"Removing old backup of partial solutions at {start}-{index+start-1}_codes.json")
        prev_codes_loc = os.path.join(save_dir, f"{start}-{index+start-1}_codes.json")
        if os.path.exists(prev_codes_loc):
            os.remove(prev_codes_loc)

def load_problems_and_solutions(test_loc, save_dir, start, end, logger):
    """
    Loads all problems in the range start-end from the APPS dataset.
    Loads all solutions generated so far.
    Checks if partial solution for the next problem to attempt exists.
    If that's the case, loads also partial solution and skips that problem.
    Return remaining problems and solutions so far.
    """
    def find_matching_string(strings, start):
        for string in strings:
            if string.startswith(start) and string.endswith("_codes.json"):
                return string
        return None

    # Load APPS problems in the start-end range
    problems = load_problems(test_loc, start, end)
    files_in_dir = os.listdir(save_dir)
    if len(files_in_dir) > 0:
        # Load solutions generated so far '{start}-..._codes.json'
        codes_loc = find_matching_string(files_in_dir, f"{start}-")
        if codes_loc is not None:
            logger.info(f"Loading solutions from {codes_loc}")
            with open(os.path.join(save_dir, codes_loc), "r") as f:
                solutions = json.load(f)
        else:
            logger.info(f"No pre-existing solutions found.")
            solutions = {}
    else:
        logger.info(f"No pre-existing solutions found.")
        solutions = {}
 
    index = len(solutions)

    # Check if problem index+1 has a partial solution
    if os.path.exists(os.path.join(save_dir, f"{index+1}")):
        logger.info(f"Loading partial solution for problem {index+1}")
        with open(os.path.join(save_dir, f"{index+1}", "best_code_rollout.py"), "r") as f:
            best_code_rollout = f.read()
        solutions[index+1] = [best_code_rollout]
        index += 1

    problems = problems[index:]
    return problems, solutions  