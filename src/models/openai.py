import os
import sys
import openai

# Import src code from parent directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import src.code_helpers as code_helpers

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

class OpenAILLM():
    def __init__(self, openai_api_key_path, openai_org_id_path, model='gpt-3.5-turbo', logger=DEFAULT_LOGGER):
        self.model = code_helpers.get_model_name(model)
        
        with open(openai_api_key_path, "r") as f:
            OPENAI_API_KEY = f.read().strip()
        
        with open(openai_org_id_path, "r") as f:
            OPENAI_ORG_ID = f.read().strip()
            
        client = openai.Client(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORG_ID
        )
        
        self.client = client

        self.logger = logger
        self.logger.info(f"OpenAI API client loaded for model {self.model}.")
        
    def get_completion(self, prompt, exclude_prompt=False, **kwargs):
        messages = code_helpers.get_messages(prompt, splits=["system", "user", "assistant"])

        max_tokens = kwargs.pop("max_new_tokens", 4096)
        # Should this be moved from code helpers to this class?
        completion = code_helpers.get_completion(self.client, messages, model=self.model, max_tokens=max_tokens, **kwargs)
        if messages[-1]["role"] == "assistant":
            completion = messages[-1]["content"] + "\n" + completion
            
        return completion