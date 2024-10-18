import numpy as np

class MockLLM:
    def __init__(self, code_completions, seed=None):
        self.code_completions = code_completions
        self.np_random = np.random.RandomState(seed)
    
    def set_seed(self, seed):
        self.np_random.seed(seed)
    
    def get_completion(self, prompt, exclude_prompt=False, **kwargs):
        return f'## Correct code\n```python\n{self.np_random.choice(self.code_completions)}```'