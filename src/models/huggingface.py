import os
import sys
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import src.code_helpers as code_helpers

class HuggingFaceLLM():
    def __init__(self, model_name, cache_dir=None, logger=DEFAULT_LOGGER, **kwargs):
        self.model_name = code_helpers.get_model_name(model_name)
        self.logger = logger
        self.logger.info(f"Loading model {self.model_name}...")
        token = os.environ.get("HF_TOKEN", None) # Token needed for gated models (Llama 2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Using bfloat16 results in RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16' with Deepseek Coder
        # (https://github.com/huggingface/diffusers/issues/3453)
        # Float16 on the other hand can result in RuntimeError: probability tensor contains either `inf`, `nan` or element < 0, likely due to overflow
        # (https://github.com/meta-llama/llama/issues/380)
        # self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if 'mistralai' in self.model_name or 'llama' in self.model_name:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        
        # TODO this currently is not working, we need to figure out how to use flash attention on AMD GPUs
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=AMD
        use_flash_attention = kwargs.get("use_flash_attention", False)
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        self.logger.debug(f"Using flash attention: {use_flash_attention}")
        
        if cache_dir is None:
            cache_dir = os.environ.get("HF_HOME", None)
        self.logger.debug(f"Huggingface cache dir: {cache_dir}")

        # Tokenizer params
        padding_side = kwargs.get("padding_side", "left")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=self.dtype, token=token, padding_side=padding_side)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=self.dtype, token=token, device_map='auto', attn_implementation=attn_implementation)
        self.model.eval()
        
        self.logger.info(f"Model {self.model_name} loaded on device {self.device} with {self.dtype} precision.")

        # WizardCoder extra token fix
        # https://github.com/huggingface/transformers/issues/24843
        if "WizardLM" in self.model_name:
            special_token_dict = self.tokenizer.special_tokens_map
            self.tokenizer.add_special_tokens(special_token_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.logger.debug(f"Special tokens added to tokenizer: {special_token_dict}")

        # Generation params
        self.max_new_tokens = kwargs.get("max_new_tokens", 1500)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 100)
        self.top_p = kwargs.get("top_p", 0.8)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.num_beams = kwargs.get("num_beams", 1)

        self.logger.debug(f"Generation params: max_new_tokens={self.max_new_tokens}, min_new_tokens={self.min_new_tokens}, temperature={self.temperature}, top_k={self.top_k}, top_p={self.top_p}, num_return_sequences={self.num_return_sequences}, num_beams={self.num_beams}")

        # Possibly switch token to unk_token to avoid issues? (https://github.com/meta-llama/llama/issues/380)
        self.tokenizer.pad_token = kwargs.get("tokenizer_pad", self.tokenizer.eos_token)
        self.tokenizer.padding_side = kwargs.get("tokenizer_padding_side", "left")

    def get_completion(self, prompt, exclude_prompt=True, **kwargs):
        start = time.perf_counter()
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        merge_system = True if "mistralai" in self.model_name.lower() else False # Fix for Mixtral as it doesn't support system messages
        messages = code_helpers.get_messages(prompt, merge_system=merge_system)
        self.logger.debug(f"Messages: {messages}")
        last_message = messages[-1]
        if last_message["role"] == "assistant":
            add_generation_prompt = False
        else:
            add_generation_prompt = True
        
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, return_dict=True, return_tensors="pt").to(self.device)
        self.logger.debug(f"Input tokens: {self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}")
        self.logger.debug(f"Number of input tokens: {len(inputs['input_ids'][0])}")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, min_new_tokens=self.min_new_tokens, 
                                      do_sample=True, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, 
                                      num_return_sequences=self.num_return_sequences, num_beams=self.num_beams, pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        self.logger.debug(f"Number of new output tokens: {len(outputs[0]) - len(inputs[0])}")
        completion = self.tokenizer.decode(outputs[0][len(inputs[0]):] if exclude_prompt else outputs[0], skip_special_tokens=True)
        
        if exclude_prompt and last_message["role"] == "assistant":
            completion = last_message["content"] + "\n" + completion
        end = time.perf_counter()
        self.logger.debug(f"Completion generated in {end-start:.2f}s")

        return completion