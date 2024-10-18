import os
import sys
import json
import time
import numpy as np
import traceback
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models import OpenAILLM, HuggingFaceLLM
from src.world_coder.world_coder_cwm import WorldCoderSimulatorLLM, Parser, WorldCoder
from src.replay_buffer.fill_replay_buffer import fill_buffer
import src.environments.rtfm.rtfm_utils as rtfm_utils

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
    
def is_continuous_state_space(env):
    """
    Check if the environment has a continuous state space.
    """
    discrete_envs = ['Blackjack-v1', 'CliffWalking-v0', 'Taxi-v3']
    return env.strip() not in discrete_envs

def initialize_simulator_llm(LLM, parser, env, logger=None):
    example_dir = None
    PROMPTS_PATH = os.path.join(PROJECT_ROOT, "data", "prompts", "gymnasium_envs" if env != "rtfm" else "rtfm")
    generate_prompt_path = os.path.join(PROMPTS_PATH, env, "generate.md")
    fix_prompt_path = os.path.join(PROMPTS_PATH, env, "debug.md")
    improve_prompt_path = os.path.join(PROMPTS_PATH, env, "improve.md")
    
    if env == "rtfm":
        example_dir = os.path.join(PROJECT_ROOT, "data", "rtfm_examples")
        generate_prompt_path = os.path.join(PROMPTS_PATH, "generate.md")
        fix_prompt_path = os.path.join(PROMPTS_PATH, "fix.md")
        improve_prompt_path = os.path.join(PROMPTS_PATH, "improve.md")

        flags = rtfm_utils.Flags(env="groups_simple_stationary-v0")
        gym_env = rtfm_utils.create_env(flags)
        env = rtfm_utils.TextRTFM(gym_env)

        n_transitions = 1000
        file_path = os.path.join(PROJECT_ROOT, "data", "replay_buffers", "rtfm", "simple_stationary")
        train_buffer = fill_buffer(env, capacity=n_transitions, file_path=file_path, buffer_name='train_buffer', force_new=False)
        transitions = train_buffer.sample(n_transitions, sample_tensor=False)
        check_valid_actions = True
        continuous_space = False
    else:
        buffer_path = os.path.join(PROJECT_ROOT, "data", "replay_buffers", "gymnasium_envs", env)
        logger.info("Buffer path: ", buffer_path)
        train_buffer = fill_buffer(env=None, capacity=None, file_path=buffer_path, buffer_name="train_buffer", force_new=False)
        # Load all transitions.
        transitions = train_buffer.sample(batch_size=None, sample_tensor=False)
        check_valid_actions = False
        continuous_space = is_continuous_state_space(env)
        print(f"Continuous space: {continuous_space}")
    
    simulator_llm = WorldCoderSimulatorLLM(LLM, parser, env, transitions, generate_prompt_path, fix_prompt_path, improve_prompt_path, example_dir, check_valid_actions, continuous_state=continuous_space, logger=logger)
    return simulator_llm

def generate_solutions(
    model,
    env,
    hf_cache=None,
    budget=1,
    save_dir="results/cwm/default_exp",
    log_params={"log_file":True,"log_level":"INFO"},
    **kwargs
):
    """
    Processes a part of the APPS dataset and generates LLM solutions to the assigned problems.
    The problems are then wrote to file as '{start}-{end}_codes.json'.
    """
    # Create dir where to save generated solutions
    save_dir = os.path.join(save_dir, env)
    create_directory(save_dir) 
        
    # Init logger used in WorldCoder
    logger = initialize_logging(save_dir, **log_params)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results will be saved in {save_dir}")
    
    # Init LLM 
    LLM = initialize_model(model, PROJECT_ROOT, hf_cache, logger, **kwargs)
    
    # Init parser (handles parsing of LLM output into well-formatted code)
    parser = Parser()
    
    # Store solutions in format problem_id:[sol_1,...,sol_n] ; n=1 in this script
    solutions = {}
    
    start_time = time.time()
    prob_start_t = time.time()
    # Prepare problem-dependent simulator_llm for WorldCoder
    simulator_llm = initialize_simulator_llm(LLM, parser, env, logger=logger)
    
    select_sequential = kwargs.get("select_sequential", False) # Use sequential selection for ablation
    
    # Init WorldCoder instance to run code generation
    world_coder = WorldCoder(simulator_llm, logger=logger, select_sequential=select_sequential)
    
    try:
        # Run WorldCoder
        best_code_rollout, highest_value = world_coder.run(num_simulations=budget)
        if best_code_rollout is None:
            # This happens if all codes during generation score 0.0 or are bugged
            best_code_rollout = "return"
        assert isinstance(best_code_rollout, str), "Bad return from worldcoder.run. Best code rollout should be a string!"
        
        # Print resulting tree structure
        logger.info("Best code rollout:\n",best_code_rollout)
        logger.info("Highest value:", highest_value)

    except Exception as e:
        print(f"Code completion failed with error {e}")
        print(traceback.format_exc())
        best_code_rollout = "return"
        highest_value = 0.0
        
    solutions[env] = {
        "solution": best_code_rollout,
        "value": highest_value
    }
    programs = world_coder.programs
    print([p.results for p in programs])
    value_list = [p.results['avg_reward'] for p in programs]
    solutions[env]["value_list"] = value_list
    
    prob_tot_t = time.time() - prob_start_t
    logger.info(f"Environment {env} took {int(prob_tot_t // 60)} minutes and {int(prob_tot_t % 60)} seconds")
    
    total_time = time.time() - start_time
    logger.info(f"Total time: {int(total_time // 60)} minutes and {int(total_time % 60)} seconds")
        
    # Save solutions to file start-end_codes.json for parallel generation across jobs
    codes_loc = os.path.join(save_dir, f"worldcoder.json")
    with open(codes_loc, "w+") as f:
        json.dump(solutions, f)