import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.code_helpers import load_py_files
from src.code_helpers import save_tree
from src.models import OpenAILLM, HuggingFaceLLM, MockLLM
from src.replay_buffer.fill_replay_buffer import fill_buffer
import src.environments.rtfm.rtfm_utils as rtfm_utils

from src.mcts import GIF_MCTS
from src.mcts.gif_mcts import Parser, SimulatorLLMCWM
from src.mcts.mcts_visualization import print_tree

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
    
    simulator_llm = SimulatorLLMCWM(LLM, parser, env, transitions, generate_prompt_path, fix_prompt_path, improve_prompt_path, example_dir, check_valid_actions, continuous_state=continuous_space, logger=logger)
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

def generate_environment(
    model,
    env,
    hf_cache=None,
    budget=1,
    save_dir="results/cwm/default_exp",
    mcts_params={},
    log_params={"log_file":True,"log_level":"INFO"},
    **kwargs
):
    """
    Processes one of the CWM benchmark environments and generates an LLM solution to the assigned problem.
    """
    # Create dir where to save generated solutions
    create_directory(save_dir) 
        
    # Init logger used in GIF-MCTS
    logger = initialize_logging(save_dir, **log_params)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results will be saved in {save_dir}")
    logger.info(f"Running environment: {env}")
    
    # Get and finalize params for mcts
    if mcts_params is None:
        mcts_params = get_default_mcts_params()
    mcts_params['logger'] = logger
    mcts_params['save_path'] = save_dir
    
    # Init LLM 
    LLM = initialize_model(model, PROJECT_ROOT, hf_cache, logger, **kwargs)
    
    # Init parser (handles parsing of LLM output into well-formatted code)
    parser = Parser()
    
    start_time = time.time()
    
    # Prepare problem-dependent simulator_llm for gif-mcts
    simulator_llm = initialize_simulator_llm(LLM, parser, env, logger=logger)
    
    # Init MCTS_LLM instance to run code generation
    mcts_llm = GIF_MCTS(simulator_llm, **mcts_params)
    
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
        save_results(save_dir, env, best_code_rollout, highest_value, extra_info)

    except Exception as e:
        print(f"Code completion failed with error {e}")
        best_code_rollout = "return"
        
    
    total_time = time.time() - start_time
    logger.info(f"Environment {env} completed in {total_time:.2f} seconds")