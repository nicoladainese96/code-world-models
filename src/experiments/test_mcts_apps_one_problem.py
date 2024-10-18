import os
import sys
import fire
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.code_helpers import load_py_files
from src.code_helpers import save_tree
from src.models import OpenAILLM, HuggingFaceLLM, MockLLM

from src.mcts import GIF_MCTS
from src.mcts.gif_mcts import Parser, SimulatorLLMApps
from src.mcts.mcts_visualization import print_tree

from mloggers import ConsoleLogger, FileLogger, MultiLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def save_results(run_dir, best_code_rollout, highest_value, extra_info):
    tree_out_dir = os.path.join(run_dir, "tree")
    root = extra_info['root']
    mcts_state = {
        'best_code_rollout': best_code_rollout,
        'highest_value': highest_value,
        'extra_info': extra_info,
    }
    save_tree(tree_out_dir, root, prefix='', save_python_files=True, mcts_state=mcts_state)

def main(
    model="gpt-3.5-turbo",
    hf_cache=None,
    budget=1,
    code_block_length=5,
    total_code_length=1000,
    eps=1,
    ucb_c=0.1,
    discount=1.0,
    v_g_new_init=0.5,
    g_counts_init=2,
    v_f_new_init=0.5,
    f_counts_init=2,
    v_i_new_init=0.55, # to be tuned
    i_counts_init=2, # to be tuned
    fixes=3,
    log_file=True,
    log_level="INFO",
    output_dir=os.path.join(PROJECT_ROOT, 'runs'),
    **kwargs
):
    create_output_directory(output_dir)
    run_id = os.environ.get("SLURM_JOB_ID", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir)
    
    logger = initialize_logging(run_dir, log_file, log_level)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results will be saved in {run_dir}")
    
    LLM = initialize_model(model, PROJECT_ROOT, hf_cache, logger, **kwargs)
    
    parser = Parser()
    
    APPS_PATH = os.path.join(PROJECT_ROOT, "data", "APPS", "test")
    prob_path = os.path.join(APPS_PATH, '3000')
    simulator_llm = initialize_simulator_llm(LLM, parser, prob_path, logger=logger)

    mcts_params = dict(
        node_length=code_block_length,
        ucb_c=ucb_c,
        discount=discount,
        max_actions=int(total_code_length/code_block_length),
        eps=eps,
        v_g_new_init=v_g_new_init,
        g_counts_init=g_counts_init,
        v_f_new_init=v_f_new_init,
        f_counts_init=f_counts_init,
        v_i_new_init=v_i_new_init,
        i_counts_init=i_counts_init,
        logger=logger,
        max_fix_chain_length=fixes,
    )
    mcts_llm = GIF_MCTS(simulator_llm, **mcts_params)
    
    B = budget
    best_code_rollout, highest_value, extra_info = mcts_llm.run(num_simulations=B)
    
    logger.info("Best code rollout:\n",best_code_rollout)
    logger.info("Highest value:", highest_value)
    print_tree(extra_info, logger=logger)
    
    save_results(run_dir, best_code_rollout, highest_value, extra_info)
    
if __name__ == "__main__":
    fire.Fire(main)
