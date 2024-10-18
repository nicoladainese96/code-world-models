import os
import sys
import json
import time
import random
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)
logger = DEFAULT_LOGGER

import warnings
warnings.filterwarnings("default")
warnings.showwarning = lambda *args, **kwargs: logger.warning(str(args[0]))

from src.models import HuggingFaceLLM, OpenAILLM
from src.apps.helpers import generate_prompt, generate_CoT_prompt, generate_Plan_and_Solve_prompt, load_problems
import src.code_helpers as code_helpers
from src.apps.helpers import load_problems_and_solutions, backup_solutions  

def get_code_completion(model, prompt, logger, **kwargs):
    # Get (unfiltered) model completion
    completion = model.get_completion(prompt, **kwargs)
    logger.info(completion)

    # Get code-only completion
    code_completion = code_helpers.extract_code(completion)
    logger.info(code_completion)
    
    return code_completion

def get_mock_code_completion(solutions_path):
    # Read one example solution
    if os.path.exists(solutions_path):
        with open(solutions_path, 'r') as f:
            sols = json.load(f)
        sample_sol = random.choice(sols)
    else:
        sample_sol = 'return'
    return sample_sol

def generate_solutions(
    model,
    hf_cache=None,
    start=0,
    end=None,
    mock=False,
    dataset_path="data/APPS/test",
    test_loc="data/APPS/test.json",
    save_dir="./results/apps/default_exp",
    strategy='vanilla-zero-shot',
    sols_per_prob=1,
    **kwargs
):
    """
    Processes a part of the APPS dataset and generates LLM solutions to the assigned problems.
    The problems are then wrote to file as '{start}-{end}_codes.json'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    logger = DEFAULT_LOGGER
    if mock:
        model = None
    elif "gpt" in model:
        OPENAI_API_KEY_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_key")
        OPENAI_ORG_ID_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_org")
        model = OpenAILLM(OPENAI_API_KEY_PATH, OPENAI_ORG_ID_PATH, model=model, logger=logger)
    else:
        model = HuggingFaceLLM(model, cache_dir=hf_cache, logger=logger, **kwargs)
    
    # Return missing problems and solutions so far
    problems, solutions = load_problems_and_solutions(test_loc, save_dir, start, end, logger=logger)
    offset = len(solutions)
    logger.info(f"Problems to solve: {len(problems)}")
    logger.info(f"Problems paths: {problems}")
    logger.info(f"Problems solved so far: {len(solutions)}")
    
    start_time = time.time()
    for idx, prob_path in enumerate(tqdm(problems)):
        index = idx + offset
        # Prepare input
        prompt_path = os.path.join(prob_path, 'question.txt')
        test_case_path = os.path.join(prob_path, "input_output.json")
        if strategy == "vanilla-zero-shot":
            prompt = generate_prompt(test_case_path, prompt_path)
        elif strategy == 'zero-shot-CoT':
            prompt = generate_CoT_prompt(test_case_path, prompt_path)
        elif strategy == 'plan-and-solve':
            prompt = generate_Plan_and_Solve_prompt(test_case_path, prompt_path)
        else:
            raise NotImplementedError
        logger.info(f"Prompt: {prompt}")
        
        code_completions = []
        for i in range(sols_per_prob):
            if mock:
                # Copy-paste ground truth solution
                solutions_path = os.path.join(prob_path, "solutions.json")
                code_completion = get_mock_code_completion(solutions_path)
            else:
                try:
                    code_completion = get_code_completion(model, prompt, logger, **kwargs)
                except Exception as e:
                    print(f'Code completion failed with error {e}')
                    code_completion = 'return'

            code_completions.append(code_completion)

        solutions[index+start] = code_completions
        backup_solutions(solutions, start, index, save_dir, logger)

    total_time = time.time() - start_time
    logger.info(f"Total time: {int(total_time // 60)} minutes and {int(total_time % 60)} seconds")
        
        
    codes_loc = os.path.join(save_dir, f"{start}-{end}_codes.json")
    with open(codes_loc, "w") as f:
        json.dump(solutions, f)
