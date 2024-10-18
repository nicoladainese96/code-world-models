import os
import re
import sys
import json
import time
import signal
import cProfile
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.code_helpers import load_py_files
from src.code_helpers import save_tree
from src.models import OpenAILLM, HuggingFaceLLM, MockLLM

from src.mcts import GIF_MCTS
from src.mcts.gif_mcts import Parser, SimulatorLLMApps
from src.mcts.mcts_visualization import print_tree
from src.apps.helpers import load_problems_and_solutions, backup_solutions  

from mloggers import ConsoleLogger, FileLogger, MultiLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)
logger = DEFAULT_LOGGER

def create_directory(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)
        
def initialize_logging(run_dir, log_file=False, log_level="INFO"):
    if log_file:
        log_path = os.path.join(run_dir, "log.json")
        loggers = [ConsoleLogger(LogLevel[log_level]), FileLogger(log_path, LogLevel[log_level])]
        logger = MultiLogger(loggers)
    else:
        logger = DEFAULT_LOGGER
        
    import warnings
    warnings.filterwarnings("default")
    warnings.showwarning = lambda *args, **kwargs: logger.warning(str(args[0]))
    
    return logger

def initialize_model(model, PROJECT_ROOT, hf_cache=None, logger=None, **kwargs):
    """
    Initialize an LLM instance, either from OpenAI or HuggingFace.
    
    Inputs:
    -------
    model: str, model name
    PROJECT_ROOT: should be path/to/CodeWorldModels
    hf_cache: dir where model weights are stored
    logger: logger instance
    **kwargs: args to be passed to the HuggingFaceLLM
    
    Returns:
    --------
    LLM instance (see src.models for more info)
    """
    if "gpt" in model:
        OPENAI_API_KEY_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_key")
        OPENAI_ORG_ID_PATH = os.path.join(PROJECT_ROOT, "openai", "openai_org")
        return OpenAILLM(OPENAI_API_KEY_PATH, OPENAI_ORG_ID_PATH, model=model, logger=logger)
    elif "mock" in model: # this doesn't work for APPS!!
        completions = load_py_files(os.path.join(PROJECT_ROOT, 'data', 'trees', 'rtfm', 'gpt-4-turbo-preview', 'simple_stationary_1'))
        return MockLLM(completions, seed=0)
    else:
        return HuggingFaceLLM(model, cache_dir=hf_cache, logger=logger, **kwargs)
    
def initialize_simulator_llm(LLM, parser, prob_path, logger=None):
    PROMPTS_PATH = os.path.join(PROJECT_ROOT,"data","prompts","apps")
    generate_prompt_path = os.path.join(PROMPTS_PATH,"generate.md")
    fix_prompt_path = os.path.join(PROMPTS_PATH,"fix.md")
    improve_prompt_path = os.path.join(PROMPTS_PATH,"improve.md")
    simulator_llm = SimulatorLLMApps(LLM, parser, prob_path, generate_prompt_path, fix_prompt_path, improve_prompt_path, logger=logger)
    return simulator_llm

def save_results(run_dir, prob_id, best_code_rollout, highest_value, extra_info):
    """
    Save results of the GIF-MCTS for code generation to file.
    """
    tree_out_dir = os.path.join(run_dir, str(prob_id), "tree")
    create_directory(tree_out_dir)
    
    root = extra_info['root']
    mcts_state = {
        'best_code_rollout': best_code_rollout,
        'highest_value': highest_value,
        'extra_info': extra_info,
    }
    save_tree(tree_out_dir, root, prefix='', save_python_files=False, mcts_state=mcts_state)
    
def get_default_mcts_params():
    """
    Get default GIF-MCTS params if no params are received in input.
    """
    mcts_params = dict(
        node_length=5,
        ucb_c=0.1,
        discount=1.0,
        max_actions=int(1000/5),
        eps=1,
        v_g_new_init=0.5,
        g_counts_init=2,
        v_f_new_init=0.5,
        f_counts_init=2,
        v_i_new_init=0.55,
        i_counts_init=2, 
    )
    return mcts_params

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
    mcts_params={},
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
        
    # Init logger used in GIF-MCTS
    logger = initialize_logging(save_dir, **log_params)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results will be saved in {save_dir}")
    
    # Get and finalize params for mcts
    if mcts_params is None:
        mcts_params = get_default_mcts_params()
    mcts_params['logger'] = logger
    
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
        logger.info(f"Problem {index+start}: {prob_path}")

        prob_start_t = time.time()
        # Prepare problem-dependent simulator_llm for gif-mcts
        simulator_llm = initialize_simulator_llm(LLM, parser, prob_path, logger=logger)
        
        # Init MCTS_LLM instance to run code generation
        mcts_llm = GIF_MCTS(simulator_llm, save_path=os.path.join(save_dir, f"{index+start}"), **mcts_params)
        
        try:
            # Run GIF-MCTS
            best_code_rollout, highest_value, extra_info = mcts_llm.run(num_simulations=budget)
            if best_code_rollout is None:
                # This happens if all codes during generation score 0.0 or are bugged
                best_code_rollout = "return"
            assert isinstance(best_code_rollout, str), "Bad return from mcts_llm.run. Best code rollout should be a string!"
            
            # Print resulting tree structure
            logger.info("Best code rollout:\n",best_code_rollout)
            logger.info("Highest value:", highest_value)
            print_tree(extra_info, logger=logger)

            # Save all results to file -> run_dir/prob_id/tree
            save_results(save_dir, index+start, best_code_rollout, highest_value, extra_info)
    
        except Exception as e:
            logger.info(f"Code completion failed with error {e}")
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
        
    do_profile = os.environ.get("PROFILE", False)
    if do_profile:
        dump_profile()