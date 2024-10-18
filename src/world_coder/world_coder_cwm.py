import os
import sys
import numpy as np

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import src.code_helpers as code_helpers
#from src.mcts.mcts_code_generation_apps import SimulatorLLM
from src.mcts import SimulatorLLMCWM as SimulatorLLM

class Program():
    def __init__(self):
        self.total_code = '' # just for legacy reasons
        self.full_code = ''
        self.extra_info = {}

    def expand(self, code, results, extra_info, C):
        self.full_code += code
        self.results = results

        self.extra_info = extra_info
        self.bug = extra_info['bug']
        
        # Init Beta prior
        self.alpha_p = 1 + results['avg_reward']*C
        self.beta_p = 1 + (1 - results['avg_reward'])*C
        
    def sample_expected_reward(self):
        sample = np.random.beta(self.alpha_p, self.beta_p) 
        return sample

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
    
    def parse(self, completion, action_type):
        self.logger.info("Completion:\n", completion)
        
        if action_type == 'improve' or action_type == 'fix':
            new_code = self.get_new_code(completion) # Cut away the critique part, which might contain code snippets
            new_code = code_helpers.extract_code(new_code)
        else:
            new_code = code_helpers.extract_code(completion)

        return new_code
    
class WorldCoderSimulatorLLM(SimulatorLLM):
    def step(self, state_node, action_type, highest_value, best_code_rollout):
        prompt = self.form_prompt(state_node, action_type, highest_value, best_code_rollout) # ok
        self.logger.info("Prompt: \n", prompt)
        # Here we expect only the new part of the code and not the one included in the prompt
        # This is expected to be until the end of the program
        extra_info = {}
        completion = self.LLM.get_completion(prompt) 
        
        full_program = self.parser.parse(completion, action_type)
        
        # Full program is returned because we try to fix the indentation within the method
        # extra info has 'bug', 'exception' and 'misclassified_transition' that are needed for further refinements
        full_program, prediction_success_rate, extra_info = self.check_sampled_code(full_program)
        results = {
            'avg_reward': prediction_success_rate,
            'strict_reward':float(prediction_success_rate == 1.0)
        }
        
        return full_program, results, extra_info

class WorldCoder():
    def __init__(self, 
                 simulator, 
                 logger,
                 select_sequential=False,
                 C=5,
        ):
        # Internal classes
        self.simulator = simulator
        self.logger = logger

        # internal constants and parameters
        self.C = C
        self.max_value = 1.0

        # Internal variables
        self.programs = []
        self.highest_value = 0.0
        self.best_code_rollout = None
        self.select_sequential = select_sequential
        self.logger.info("Sequential selection: ", self.select_sequential)

    def run(self, num_simulations):
        
        program, results = self.generate_seed_program() # 1 call to generate_seed_program

        self.highest_value = results['avg_reward']
        self.best_code_rollout = program.full_code
        
        if self.highest_value == self.max_value:
            self.logger.info("Correct solution found!")
            self.logger.info('Value: ', self.highest_value)
            return self.best_code_rollout, self.highest_value

        for i in range(num_simulations-1): # num_simulations - 1 calls left from budget
            # Thompson Sampling for selecting the program
            program = self.select_program() if not self.select_sequential else self.select_program_sequential()

            # LLM-call to either fix or improve the selected program and run it against test cases to estimate reward
            refined_program, results = self.refine_program(program)

            if results['avg_reward'] > self.highest_value:
                self.highest_value = results['avg_reward']
                self.best_code_rollout = refined_program.full_code

            if self.highest_value == self.max_value:
                self.logger.info("Correct solution found!")
                self.logger.info('Value: ', self.highest_value)
                return self.best_code_rollout, self.highest_value
            
            self.update_prior(program, results) # usually just increase beta term of 1 if strict_reward != 1.0
        
        return self.best_code_rollout, self.highest_value
    
    def generate_seed_program(self):
        root_program = Program()
        new_code, results, extra_info = self.simulator.step(root_program, 'generate', self.highest_value, self.best_code_rollout)
        root_program.expand(new_code, results, extra_info, self.C)
        self.programs.append(root_program)
        return root_program, results

    def refine_program(self, program):
        if program.bug:
            action_type = 'fix'
        else:
            action_type = 'improve'
        new_code, results, extra_info = self.simulator.step(program, action_type, self.highest_value, self.best_code_rollout)   
        refined_program = Program()
        refined_program.expand(new_code, results, extra_info, self.C)
        self.programs.append(refined_program)
        return refined_program, results

    def select_program(self):
        # Sample expected rewards for each program from Beta distributions
        expected_rewards = [program.sample_expected_reward() for program in self.programs]
        # Select the program with the highest expected reward for next refinement
        selected_idx = np.argmax(expected_rewards)
        selected_program = self.programs[selected_idx]
        return selected_program
    
    def select_program_sequential(self):
        # Select the last program for refinement
        # Used for ablation study
        selected_program = self.programs[-1]
        return selected_program

    def update_prior(self, program, results):
        # Update the Beta prior of the selected program
        # alpha_p is like the number of successes and beta_p is like the number of failures
        # strict_reward is 1 if the program is correct and 0 otherwise
        program.alpha_p += results['strict_reward']
        program.beta_p += 1 - results['strict_reward']

    