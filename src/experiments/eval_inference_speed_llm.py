import os
import sys
import fire
import json
import time
import autopep8
import numpy as np
from mloggers import ConsoleLogger, LogLevel
from torch import device
from importlib import reload

DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from src.models import OpenAILLM, HuggingFaceLLM
from src.replay_buffer import ReplayBuffer

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
    else:
        return HuggingFaceLLM(model, cache_dir=hf_cache, logger=logger, **kwargs)
    
def format_prompt_example(obs, action):
    example = f"### Observation\n{obs}\n\n### Action\n{action}\n"
    return example

def extract_completion(completion):
    next_obs = completion.split("### Next Observation ###")[1].strip()
    next_obs = next_obs.split('[')[1].split(']')[0].replace('\n', '').replace(',', '')    
    next_obs = next_obs.split()
    next_obs = [float(x) for x in next_obs if x]
    next_obs = np.array(next_obs, dtype=np.float32)

    reward = np.float16(completion.split("Reward:")[1].split("\n")[0].strip())
    done = completion.split("Done:")[1].split("\n")[0].strip() == 'True'
    
    return next_obs, reward, done
    
    return next_obs, reward, done

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
    
def load_cwm(code_path):
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            full_program = f.read()

        # Temporarily write code as importable gen_code_world_model module
        full_program = autopep8.fix_code(full_program)
        with open('gen_code_world_model.py', 'w') as f:
            f.write(full_program)

        try:
            # Import generated code
            import gen_code_world_model

            # Force reloading of the module
            gen_code_world_model = reload(gen_code_world_model)

            # Generate an instance of the Environment class
            code_env = gen_code_world_model.Environment()
            return code_env
        except Exception as e:
            print(f"Error loading code: {e}")
            return None

def main(
    model,
    results_dir=None,
    n_transitions=1,
    ):
    
    logger = DEFAULT_LOGGER
    model = initialize_model(model, PROJECT_ROOT, logger=logger)
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "results", "llm_prediction")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    PROMPTS_PATH = os.path.join(PROJECT_ROOT, "data", "prompts", "llm_prediction")
    envs = {"cartpole": "CartPole-v1", "humanoid": "Humanoid-v4", "half_cheetah": "HalfCheetah-v4"}
    results = {}
    for prompt_file in os.listdir(PROMPTS_PATH):
        with open(os.path.join(PROMPTS_PATH, prompt_file), 'r') as f:
            prompt = f.read()
        logger.info(f"Prompt file: {prompt_file}")
        env = envs[prompt_file.split(".")[0]]
        results[env] = []
        logger.info(f"Environment: {env}")
        logger.info(f"Prompt: {prompt}")
        
        
        file_path = os.path.join(PROJECT_ROOT, "data", "replay_buffers", "gymnasium_envs", env)
        buffer = ReplayBuffer(1000, device=device("cpu"), file_path=file_path, buffer_name='train_buffer')
        buffer.load()
        obs, actions, next_obs, rewards, dones, extras = buffer.sample(n_transitions)
        
        for i in range(n_transitions):
            obs_i = obs[i]
            action_i = actions[i]
            result = {"input": {"obs": obs_i, "action": action_i}}
            
            example = format_prompt_example(obs_i, action_i)
            logger.info(f"Example {i+1}/{n_transitions}:\n{example}")
            prompt = prompt.replace("{STATE_ACTION_TO_PREDICT}", example)
            logger.info(f"Prompt {i+1}/{n_transitions}:\n{prompt}")
            
            start_time = time.perf_counter()
            try:
                completion = model.get_completion(prompt)
                logger.info(f"Completion {i+1}/{n_transitions}:\n{completion}")
                pred_next_obs, pred_reward, pred_done = extract_completion(completion)
            except Exception as e:
                logger.error(f"Error: {e}")
                pred_next_obs, pred_reward, pred_done = None, None, None
            
            end_time = time.perf_counter() 
            logger.info(f"Time taken: {end_time - start_time}")
            result["completion"] = {"time": end_time - start_time, "text": completion}
            
            logger.info(f"Predicted next observation: {pred_next_obs}")
            logger.info(f"Predicted reward: {pred_reward}")
            logger.info(f"Predicted done: {pred_done}")
            result["prediction"] = {"next_obs": pred_next_obs, "reward": pred_reward, "done": pred_done}
            
            # Check if the prediction is correct
            next_obs_i = next_obs[i]
            reward_i = rewards[i]
            done_i = dones[i]
            
            # use np.allclose for float comparison
            try:
                result["correct"] = {
                    "next_obs": np.allclose(pred_next_obs, next_obs_i),
                    "reward": np.allclose(pred_reward, reward_i),
                    "done": pred_done == done_i
                }
                result["correct"]["all"] = all(result["correct"].values())
            except Exception as e:
                logger.error(f"Failing to compare predictions: {e}")   
                result["correct"] = {
                    "next_obs": False,
                    "reward": False,
                    "done": False,
                    "all": False
                }            
            
            # Now get prediction with CWM
            try:
                cwm_path = os.path.join(PROJECT_ROOT, "results", "gymnasium_envs", "mcts_gpt4_B10", env, "best_code_rollout.py")
                code_env = load_cwm(cwm_path)
                code_env_start_time = time.perf_counter()
                code_env.set_state(obs_i)
                next_obs_cwm, reward_cwm, done_cwm = code_env.step(action_i)
                code_env_end_time = time.perf_counter()
            except Exception as e:
                logger.error(f"Error running CWM: {e}")
                next_obs_cwm, reward_cwm, done_cwm = None, None, None
                code_env_end_time = code_env_start_time = 0
            
            logger.info(f"CWM prediction took: {code_env_end_time - code_env_start_time}")
            logger.info(f"CWM predicted next observation: {next_obs_cwm}")
            logger.info(f"CWM predicted reward: {reward_cwm}")
            logger.info(f"CWM predicted done: {done_cwm}")
            
            result["cwm_prediction"] = {
                "next_obs": next_obs_cwm,
                "reward": reward_cwm,
                "done": done_cwm,
                "time": code_env_end_time - code_env_start_time 
            }
            
            try:
                result["cwm_correct"] = {
                    "next_obs": np.allclose(next_obs_cwm, next_obs_i),
                    "reward": np.allclose(reward_cwm, reward_i),
                    "done": done_cwm == done_i
                }
                result["cwm_correct"]["all"] = all(result["cwm_correct"].values())
            except Exception as e:
                logger.error(f"Failing to compare CWM predictions: {e}")   
                result["cwm_correct"] = {
                    "next_obs": False,
                    "reward": False,
                    "done": False,
                    "all": False
                }
                
            results[env].append(result)
            
            if env == "Humanoid-v4":
                logger.info("Only running one example for Humanoid-v4")
                continue

    logger.info("Final results:")
    results = convert_to_serializable(results)
    logger.info(results)
    json.dump(results, open(os.path.join(results_dir, "results.json"), "w+"))
    
if __name__ == '__main__':
    fire.Fire(main)