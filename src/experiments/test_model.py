import os
import sys
import fire

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)

from src.models import HuggingFaceLLM
import src.code_helpers as code_helpers

def main(
    model,
    hf_cache=None,
    use_flash=False,
    **kwargs
):
    logger = DEFAULT_LOGGER
    model = HuggingFaceLLM(model, cache_dir=hf_cache, logger=logger, use_flash_attention=use_flash, **kwargs)
    PROMPTS_PATH = os.path.join(PROJECT_ROOT, "data", "prompts")
    
    generate_prompt_path = os.path.join(PROMPTS_PATH, "rtfm", "generate.md")
    with open(generate_prompt_path, "r") as f:
        prompt = f.read()
    prompt = prompt.replace(f"{{CODE}}", '')
    
    logger.info(f"Prompt: {prompt}")
    completion = model.get_completion(prompt)
    logger.info(completion)

if __name__ == "__main__":
    fire.Fire(main)