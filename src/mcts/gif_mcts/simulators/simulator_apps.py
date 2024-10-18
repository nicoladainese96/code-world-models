import os
import sys
import copy
import json
import random
import autopep8
import numpy as np

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.apps.test_one_solution import check_correctness_with_errors
import src.apps.testing_util as test_util

def compute_reward(prob_path, generation, timeout=10, debug=False, mp=True, logger=None):
    try:
        curr_res, errors = check_correctness_with_errors(prob_path, generation, timeout, debug, mp=mp)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
    except Exception as e:
        logger.info(f"Error in compute_reward: {e}")
        curr_res = []
        errors = [e]
        
    logger.info("Errors: ", errors)
    # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
    assert isinstance(curr_res, list)
    if len(curr_res) > 0:
        info = {"compile_error": curr_res.count(-2) / len(curr_res), "runtime_error": curr_res.count(-1) / len(curr_res), "errors":errors, "curr_res":curr_res}
    else:
        info = {"compile_error": 0, "runtime_error": 0, "errors":errors, "curr_res":curr_res}
    pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
    
    # If no compilation , but pass rate < 1, get an example of failed prediction
    if info["compile_error"] == 0.0 and pass_rate < 1.0: # and info["runtime_error"] == 0.0 
        # info['curr_res'] is a list of True and False entries, respectively indicating a passed or failed unit test; 
        #runtime errors are indicated with -1, while compile erorrs shouldn't be present because otherwise we wouldn't execute this part of the code
        
        # Only expose half of the tests
        # E.g. info['curr_res'] = [True, False, -1, True, False, True] yields 
        # failed_tests_indexes = [1,2]
        
        try:
            L = len(info['curr_res'])
            failed_tests_indexes = [i for i in range(L//2) if not info['curr_res'][i] or info['curr_res'][i]==-1]
            # Pick one of the failed tests
            failed_test_idx = random.choice(failed_tests_indexes)
        except:
            _input = None
            _output = None
            pred_out = None

            info['misclassified_transition'] = {
                'inputs':_input,
                'outputs':_output,
                'pred_outputs':pred_out
            }
            return pass_rate, info
        
        # load all inputs and all outputs for the problem
        if os.path.exists(os.path.join(prob_path, 'input_output.json')):
            with open(os.path.join(prob_path, 'input_output.json'), 'r') as f:
                inputs_ouptuts = json.load(f)

            # Select only the ones corresponding to the chosen failed test case
            _input = inputs_ouptuts['inputs'][failed_test_idx]
            _output = inputs_ouptuts['outputs'][failed_test_idx]

            
            if info['curr_res'][failed_test_idx]==-1:
                # Here we already know which runtime error we encountered
                pred_out = str(info['errors'][failed_test_idx])
                
            else:
                # Run again the test only for them and get the predicted output
                pred_out = test_util.run_test_with_errors(
                    prob_path=prob_path, 
                    test=generation, 
                    debug=False, 
                    timeout=4, # 4 seconds, default used inside test_util.run_test_with_errors
                    test_idx=failed_test_idx
                )
        else:
            pass
    else:
        # In case of: compile error, runtime error, perfect solution, no input_output.json found, default to:
        _input = None
        _output = None
        pred_out = None

    info['misclassified_transition'] = {
        'inputs':_input,
        'outputs':_output,
        'pred_outputs':pred_out
    }
    return pass_rate, info

class SimulatorLLM():
    def __init__(self, LLM, parser, prob_path, generate_prompt_path, fix_prompt_path, improve_prompt_path, logger=DEFAULT_LOGGER):
        self.LLM = LLM
        self.parser = parser
        self.prob_path = prob_path
        # Not usedb
        self.g_prompt_path = generate_prompt_path
        self.f_prompt_path = fix_prompt_path
        self.i_prompt_path = improve_prompt_path

        self.logger = logger
        
        with open(generate_prompt_path, "r") as f:
            prompt = f.read()
        self.g_base_prompt = prompt
        
        with open(fix_prompt_path, "r") as f:
            prompt = f.read()
        self.f_base_prompt = prompt
        
        with open(improve_prompt_path, "r") as f:
            prompt = f.read()
        self.i_base_prompt = prompt
    
    def form_prompt(self, state_node, action_type, highest_value, best_code_rollout):
        if action_type == 'generate':
            return self.form_generate_prompt(state_node, highest_value, best_code_rollout)
        elif action_type == 'fix': 
            return self.form_fix_prompt(state_node, highest_value, best_code_rollout)
        else: # 'improve'
            return self.form_1_step_improve_prompt(state_node)

    def form_generate_prompt(self, state_node, highest_value, best_code_rollout):
        prompt_path = os.path.join(self.prob_path, 'question.txt')
        with open(prompt_path, "r") as f:
            data = f.readlines()
            prob_description = "".join(data)

        subs = {
            "PROB_DESCRIPTION":prob_description,
            "CODE_SO_FAR": state_node.total_code,
            #"PERCENTAGE": highest_value*100,
            #"BEST_CODE_ROLLOUT": best_code_rollout
        }
        
        prompt = copy.deepcopy(self.g_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in self.g_base_prompt:
                    self.logger.warning(f"Key {key} not found in prompt file.")
                else:
                    prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
    
    def form_fix_prompt(self, state_node, highest_value, best_code_rollout):
        prompt_path = os.path.join(self.prob_path, 'question.txt')
        with open(prompt_path, "r") as f:
            data = f.readlines()
            prob_description = "".join(data)
            
        if state_node.extra_info is not None and "exception" in state_node.extra_info:
            error_msg = f"{state_node.extra_info['exception']}. Try fixing it or avoiding it for this next completion."
        else:
            error_msg = ""

        subs = {
            "PROB_DESCRIPTION":prob_description,
            #"EXAMPLES": examples, 
            "ERROR": error_msg,
            "CODE": state_node.full_code,
            #"PERCENTAGE": highest_value*100,
            #"BEST_CODE_ROLLOUT": best_code_rollout
        }
        prompt = copy.deepcopy(self.f_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in self.f_base_prompt:
                    raise ValueError(f"Key {key} not found in prompt file.")
                prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
    
    def form_1_step_improve_prompt(self, state_node):
        prompt_path = os.path.join(self.prob_path, 'question.txt')
        with open(prompt_path, "r") as f:
            data = f.readlines()
            prob_description = "".join(data)
            
        incorrect_code = state_node.full_code
        misclassified_transition = state_node.extra_info['misclassified_transition']

        subs = {
            "PROB_DESCRIPTION":prob_description,
            "CODE":incorrect_code,
            "INPUT":misclassified_transition['inputs'],
            "OUTPUT":misclassified_transition['outputs'],
            "PREDICTION":misclassified_transition['pred_outputs']
        }

        prompt = copy.deepcopy(self.i_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in prompt:
                    raise ValueError(f"Key {key} not found in prompt file.")
                prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
    
    def step(self, state_node, action_type, highest_value, best_code_rollout):
        prompt = self.form_prompt(state_node, action_type, highest_value, best_code_rollout) # ok
        self.logger.info("Prompt: \n", prompt)
        # Here we expect only the new part of the code and not the one included in the prompt
        # This is expected to be until the end of the program
        extra_info = {}
        completion = self.LLM.get_completion(prompt) 
        
        first_new_l_lines, ancestors_code, full_program, new_length, ancestors_len, critique = self.parser.parse(
            completion, state_node.total_code, state_node.node_length, action_type
        )
        
        self.logger.info("first_new_l_lines: \n", first_new_l_lines)
        if critique is not None:
            self.logger.info("Critique: \n", critique)
            
        state_node.extra_info['critique'] = critique
        
        full_program, prediction_success_rate, extra_info = self.check_sampled_code(full_program)
            
        if new_length < state_node.node_length:
            done = True
            reward = prediction_success_rate
            value = 0 
        else:
            done = False
            reward = 0
            value = prediction_success_rate
    
        return first_new_l_lines, ancestors_code, full_program, new_length, ancestors_len, reward, done, value, extra_info
    
    def check_sampled_code(self, full_program):
        self.logger.info("Full program: \n", full_program)
        try:
            full_program = autopep8.fix_code(full_program) # Can this fail?
        except Exception as e:
            self.logger.warning(f"autopep8.fix_code failed with Exception {e}. Defaulting to original full_program.")
            full_program = full_program 

        # This handles inside every compilation or runtime error
        pass_rate, extra_info = compute_reward(self.prob_path, full_program, logger=self.logger) 
        extra_info['bug'] = False
        extra_info['exception'] = None
        
        # What is a bug? Just compile error 
        # -> runtime errors are instance dependent and corresponding programs can still be good
        if extra_info["compile_error"] != 0.0: #or extra_info["runtime_error"] != 0.0:
            extra_info['bug'] = True   
            actual_errors = [e for e in extra_info['errors'] if e is not None]
            first_error = actual_errors[0] if len(actual_errors) > 0 else None
            extra_info['exception'] = first_error
            
        return full_program, pass_rate, extra_info