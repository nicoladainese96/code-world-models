import os
import sys
import json
import time
import signal
import cProfile
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models import OpenAILLM, HuggingFaceLLM
from src.world_coder.world_coder_apps import WorldCoderSimulatorLLM, Parser, WorldCoder
from src.apps.helpers import load_problems_and_solutions, backup_solutions  

from mloggers import ConsoleLogger, FileLogger, MultiLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)

def create_directory(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)

def initialize_logging(run_dir, log_file=True, log_level="INFO"):
    if log_file:
        log_path = os.path.join(run_dir, "log.json")
        loggers = [ConsoleLogger(LogLevel[log_level]), FileLogger(log_path, LogLevel[log_level])]
        logger = MultiLogger(loggers)
    else:
        logger = DEFAULT_LOGGER
    return logger

def initialize_model(model, PROJECT_ROOT, hf_cache=None, logger=None, **kwargs):
    if "gpt" in model:
        OPENAI_API_KEY_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_key")
        OPENAI_ORG_ID_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_org")
        return OpenAILLM(OPENAI_API_KEY_PATH, OPENAI_ORG_ID_PATH, model=model, logger=logger)
    else:
        return HuggingFaceLLM(model, cache_dir=hf_cache, logger=logger, **kwargs)
    
def initialize_simulator_llm(LLM, parser, prob_path, logger=None):
    PROMPTS_PATH = os.path.join(PROJECT_ROOT,"data","prompts","apps")
    generate_prompt_path = os.path.join(PROMPTS_PATH,"generate.md")
    fix_prompt_path = os.path.join(PROMPTS_PATH,"fix.md")
    improve_prompt_path = os.path.join(PROMPTS_PATH,"improve.md")
    simulator_llm = WorldCoderSimulatorLLM(LLM, parser, prob_path, generate_prompt_path, fix_prompt_path, improve_prompt_path, logger=logger)
    return simulator_llm

def dump_profile():
    PROFILE_PATH = os.path.join(PROJECT_ROOT, 'profiles')
    profiler.disable()
    job_id = os.environ.get("SLURM_JOB_ID", None)
    job_id += "_" + os.environ.get("SLURM_ARRAY_TASK_ID", None) if "SLURM_ARRAY_TASK_ID" in os.environ else ""
    print(f"Dumping profile to {os.path.join(os.getcwd(), 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    if not os.path.exists(PROFILE_PATH):
            os.makedirs(PROFILE_PATH)
    profiler.dump_stats(f"{PROFILE_PATH}/profile_{job_id if job_id is not None else 'local'}")

# Handle sigterm to dump profile if job is killed by slurm
def signal_handler(sig, frame):
    dump_profile()
    job_id = os.environ.get("SLURM_JOB_ID", None)
    job_id += "_" + os.environ.get("SLURM_ARRAY_TASK_ID", None) if "SLURM_ARRAY_TASK_ID" in os.environ else ""
    print(f"Detecting signal {sig}. Dumping profile to {os.path.join(PROJECT_ROOT, 'profiles', 'profile')}_{job_id if job_id is not None else 'local'}")
    sys.stdout.flush()
    if sig == signal.SIGTERM or sig == signal.SIGINT:
        sys.exit(1)

def generate_solutions(
    model,
    hf_cache=None,
    budget=1,
    start=0,
    end=None,
    test_loc="data/APPS/test.json",
    save_dir="results/apps/default_exp",
    log_params={"log_file":True,"log_level":"INFO"},
    **kwargs
):
    """
    Processes a part of the APPS dataset and generates LLM solutions to the assigned problems.
    The problems are then wrote to file as '{start}-{end}_codes.json'.
    """
    global profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Create dir where to save generated solutions
    create_directory(save_dir) 
        
    # Init logger used in WorldCoder
    logger = initialize_logging(save_dir, **log_params)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results will be saved in {save_dir}")
    
    # Init LLM 
    LLM = initialize_model(model, PROJECT_ROOT, hf_cache, logger, **kwargs)
    
    # Init parser (handles parsing of LLM output into well-formatted code)
    parser = Parser()
    
    # Return missing problems and solutions so far
    problems, solutions = load_problems_and_solutions(test_loc, save_dir, start, end, logger=logger)
    offset = len(solutions)
    logger.info(f"Problems to solve: {len(problems)}")
    logger.info(f"Problems paths: {problems}")
    logger.info(f"Problems solved so far: {len(solutions)}")
    
    start_time = time.time()
    for idx, prob_path in enumerate(tqdm(problems)):
        index = idx + offset
        prob_start_t = time.time()
        # Prepare problem-dependent simulator_llm for WorldCoder
        simulator_llm = initialize_simulator_llm(LLM, parser, prob_path, logger=logger)
        
        # Init WorldCoder instance to run code generation
        world_coder = WorldCoder(simulator_llm, save_path=os.path.join(save_dir, f"{index+start}"), logger=logger)
        
        try:
            # Run WorldCoder
            best_code_rollout, highest_value = world_coder.run(num_simulations=budget)
            if best_code_rollout is None:
                # This happens if all codes during generation score 0.0 or are bugged
                best_code_rollout = "return"
            assert isinstance(best_code_rollout, str), "Bad return from world_coder.run. Best code rollout should be a string!"
            
            # Print resulting tree structure
            logger.info("Best code rollout:\n",best_code_rollout)
            logger.info("Highest value:", highest_value)

        except Exception as e:
            print(f"Code completion failed with error {e}")
            best_code_rollout = "return"
            
        solutions[index+start] = [best_code_rollout]
        
        backup_solutions(solutions, start, index, save_dir, logger)
        
        prob_tot_t = time.time() - prob_start_t
        logger.info(f"Problem {index} took {int(prob_tot_t // 60)} minutes and {int(prob_tot_t % 60)} seconds")
        
    total_time = time.time() - start_time
    logger.info(f"Total time: {int(total_time // 60)} minutes and {int(total_time % 60)} seconds")
        
    # Save solutions to file start-end_codes.json for parallel generation across jobs
    codes_loc = os.path.join(save_dir, f"{start}-{end}_codes.json")
    with open(codes_loc, "w") as f:
        json.dump(solutions, f)