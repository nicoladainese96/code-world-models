import os
import sys

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

import src.code_helpers as code_helpers

class Parser():
    def __init__(self, logger=DEFAULT_LOGGER):
        self.logger = logger
    
    def get_critique(self, prompt):
        start_tag = '## Error explanation'
        end_tag = '## Correct code'
        start_idx = prompt.find(start_tag)
        end_idx = prompt.find(end_tag)
        return prompt[start_idx + len(start_tag):end_idx].strip()
    
    def get_new_code(self, prompt):
        start_tag = '## Correct code'
        start_idx = prompt.find(start_tag)
        return prompt[start_idx + len(start_tag):].strip()
    
    def parse(self, completion, ancestors_code, target_length, action_type):
        self.logger.info("Completion:\n", completion)
        
        if action_type == 'improve' or action_type == 'fix':
            critique = self.get_critique(completion)
            new_code = self.get_new_code(completion) # Cut away the critique part, which might contain code snippets
            new_code = code_helpers.extract_code(new_code)
        else:
            critique = None
            new_code = code_helpers.extract_code(completion)

            
        l = target_length
        
        if action_type == 'fix' or action_type == 'improve':
            ancestors_len = len(ancestors_code.split("\n")) 
            ancestors_code = '' # override old code
            l += ancestors_len
        else:
            # generate case 
            # we rely on the code being continued from the ancestor's code
            # this should always be the case with the new prompting strategy
            # where the assistant response starts with the ancestor's code
            ancestors_len = len(ancestors_code.split("\n")) 
            ancestors_code = '' # override old code
            l += ancestors_len

        # First extract the first l lines
        lines = new_code.split("\n")
        #lines = [line for line in lines if line.strip()] # Remove empty lines
        lines = lines[:l]
        
        # Count how many new lines are there; if less than l it means that we finished the program
        new_length = len(lines) - ancestors_len
        
        # This is the state of the child node
        first_new_l_lines = "\n".join(lines)
        
        # Also return the full executable program
        #full_program = ancestors_code+new_code
        full_program = ancestors_code+'\n'+new_code
        
        return first_new_l_lines, ancestors_code, full_program, new_length, ancestors_len, critique