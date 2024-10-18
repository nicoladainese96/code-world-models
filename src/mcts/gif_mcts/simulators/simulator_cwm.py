import os
import sys
import copy
import autopep8
import traceback
import numpy as np
from importlib import reload

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(PROJECT_ROOT)

import src.code_helpers as code_helpers

def state_matches_rtfm(x, y):
    for i in range(len(x)):
        if not np.all(x[i] == y[i]):
            return False
    return True

def check_transition_via_enum(code_env, transition, match_exact_valid_actions=False, check_valid_actions=True,
                              continuous_state=False, continuous_reward=False, weights=None):
    valid = False   
    if continuous_state:
        state_matches = lambda x, y: np.allclose(x, y, rtol=1e-05, atol=1e-05)
    else:
        state_matches = lambda x, y: state_matches_rtfm(x, y) if (type(x) == list or type(x) == tuple) else np.all(x == y)
    if continuous_reward:
        reward_matches = np.isclose
    else:
        reward_matches = lambda x, y: x == y

    if weights is None:
        weights = {
            'terminal': {'reward': 0.5, 'done': 0.5},
            # 'default': {'board': 0.125, 'inventory': 0.125, 'reward': 0.25, 'done': 0.25, 'valid_actions': 0.25}
            'default': {'next_state': 0.25, 'reward': 0.25, 'done': 0.25, 'valid_actions': 0.25}
        }
        if not check_valid_actions:
            weights['default']['valid_actions'] = 0.0
            new_total = sum(weights['default'].values())
            weights['default'] = {k: v / new_total for k, v in weights['default'].items()}
    
    transition = copy.deepcopy(transition)
    state, action, next_state, reward, done, extra_info = transition
    code_env.set_state(copy.deepcopy(state))
    if check_valid_actions:
        pred_next_state, pred_valid_actions, pred_reward, pred_done = code_env.step(action)
    else:
        pred_next_state, pred_reward, pred_done = code_env.step(action)
        pred_valid_actions = None
    if done:
        # Ignore the next state prediction, as it's not consequential
        if reward_matches(pred_reward, reward) and pred_done==done:
            valid = True
        success_rate = weights['terminal']['reward']*float(reward_matches(pred_reward, reward))+weights['terminal']['done']*float(pred_done==done)
    else:
        valid_actions = 0.0
        if not check_valid_actions or (
            (match_exact_valid_actions and len(pred_valid_actions)==len(extra_info['valid_actions']) and np.all(pred_valid_actions==extra_info['valid_actions']))
            or (not match_exact_valid_actions and len(pred_valid_actions)>0 and np.all(np.isin(pred_valid_actions, extra_info['valid_actions'])))):
            valid_actions = 1.0

        if state_matches(pred_next_state, next_state) and reward_matches(pred_reward, reward) and pred_done==done and valid_actions:
            valid = True

        if hasattr(next_state, '__len__'):
            next_state_elements = len(next_state)
            weight_next_state = weights['default']['next_state'] / next_state_elements
            next_state_success_rate = sum(
                [weight_next_state*float(state_matches(pred_next_state[i], next_state[i])) for i in range(next_state_elements)])
        else:
            next_state_success_rate = weights['default']['next_state']*float(state_matches(pred_next_state, next_state))


        success_rate = (next_state_success_rate
                        + weights['default']['reward']*float(reward_matches(pred_reward, reward))
                        + weights['default']['done']*float(pred_done==done)
                        + weights['default']['valid_actions']*valid_actions)
    
    transition_dict = {
        'state':state,
        'action':action,
        'next_state_gt':next_state,
        'valid_actions_gt':extra_info['valid_actions'],
        'reward_gt':reward,
        'done_gt':done,
        'next_state_pred':pred_next_state,
        'valid_actions_pred':pred_valid_actions,
        'reward_pred':pred_reward,
        'done_pred':pred_done,
        'success_rate':success_rate,
        'valid':valid,
        'state_matches':state_matches(pred_next_state, next_state),
        'reward_matches':reward_matches(pred_reward, reward),
        'done_matches':pred_done==done,
    }
    
    return valid, success_rate, transition_dict

def check_code_world_model(code_env, transitions, weight_reward=True, check_valid_actions=True, continuous_state=False,
                           continuous_reward=False, verbose=True):
    # Pass from (s,a,s',r,d) of torch variables of shape (batch_size,other_dims) to
    # [(s,a,s',r,d)_0,..., (s,a,s',r,d)_N] of numpy variables without the batch size

    transitions = list(zip(*transitions))
    all_valid = []
    success_rates = []
    all_misclassified_transitions = []
    
    if weight_reward:
        reward_values = np.unique(np.array([transition[3] for transition in transitions]))
        # The weight for each reward class should be n_transitions/n_transitions_in_class to give more importance to rare classes
        reward_weights = np.array([len(transitions)/len(np.where(np.array([transition[3] for transition in transitions])==reward)[0]) for reward in reward_values]) 
        reward_weights = {reward: weight for reward, weight in zip(reward_values, reward_weights)}
        reward_weights = {reward: weight/np.sum(list(reward_weights.values())) for reward, weight in reward_weights.items()}

    for transition in transitions:
        transition_valid, success_rate, transition_dict = check_transition_via_enum(
            code_env, transition, check_valid_actions=check_valid_actions, continuous_state=continuous_state,
            continuous_reward=continuous_reward
        )
        all_valid.append(transition_valid)
        success_rates.append(success_rate)
        if not transition_valid:
            all_misclassified_transitions.append(transition_dict)
    if verbose and len(all_misclassified_transitions) > 0:
        print('Misclassified transition examples:\n', all_misclassified_transitions[0])
        if len(all_misclassified_transitions) > 1:
            print('...')
            print(all_misclassified_transitions[-1])
            
    valid = np.all(all_valid)
    if weight_reward:
        success_rate = np.average(success_rates, weights=[reward_weights[transition[3]] for transition in transitions])
        print("Number of successful transitions for each reward class:", {reward: np.sum([valid and transition[3]==reward for valid, transition in zip(all_valid, transitions)]) for reward in reward_values}) if verbose else None
    else:
        success_rate = np.mean(success_rates)
        
    return valid, all_valid, success_rates, success_rate, all_misclassified_transitions

def get_and_format_transition_input(transition, rtfm_format=False):
    out = ""
    if rtfm_format:
        out += f"Initial state:\n{transition['state'][0]}\n"
        out += f"Inventory: {transition['state'][1]}\n"
    else:
        out += f"Initial state:\n{transition['state']}\n"
    out += f"Action: {transition['action']}\n"
    return out

def get_and_format_gt_transition(transition, rtfm_format=False, check_valid_actions=True):
    out = ""
    if rtfm_format:
        out += f"Next state:\n{transition['next_state_gt'][0]}\n"
        out += f"Next inventory: {transition['next_state_gt'][1]}\n"
    else:
        out += f"Next state:\n{transition['next_state_gt']}\n"
    if check_valid_actions:
        out += f"Valid actions: {transition['valid_actions_gt']}\n"
    out += (f"Reward: {transition['reward_gt']}\n"
            f"Done: {transition['done_gt']}\n")
    return out

def get_and_format_pred_transition(transition, rtfm_format=False, check_valid_actions=True):
    out = ""
    if rtfm_format:
        out += f"Next state:\n{transition['next_state_pred'][0]}\n"
        out += f"Next inventory: {transition['next_state_pred'][1]}\n"
    else:
        out += f"Next state:\n{transition['next_state_pred']}\n"
    if check_valid_actions:
        out += f"Valid actions: {transition['valid_actions_pred']}\n"
    out += (f"Reward: {transition['reward_pred']}\n"
            f"Done: {transition['done_pred']}\n")
    return out

class SimulatorLLM():
    def __init__(self, LLM, parser, env_name, transitions, generate_prompt_path, fix_prompt_path, improve_prompt_path, example_dir, check_valid_actions=True, continuous_state=False, logger=DEFAULT_LOGGER):
        self.LLM = LLM
        self.parser = parser
        self.env_name = env_name
        self.transitions = transitions
        self.g_prompt_path = generate_prompt_path
        self.f_prompt_path = fix_prompt_path
        self.example_dir = example_dir
        self.check_valid_actions = check_valid_actions
        self.continuous_state = continuous_state

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
            prompt = self.form_generate_prompt(state_node, highest_value, best_code_rollout)
        elif action_type == 'fix': # fix
            prompt = self.form_fix_prompt(state_node, highest_value, best_code_rollout)
        else:
            prompt = self.form_1_step_improve_prompt(state_node)
        return prompt

    def form_generate_prompt(self, state_node, highest_value, best_code_rollout):
        if state_node.extra_info is not None and "exception" in state_node.extra_info:
            error_msg = f"The code you wrote raised an exception: {state_node.extra_info['exception']}. Try fixing it or avoiding it for this next completion."
        else:
            error_msg = ""

        subs = {
            #"EXAMPLES": examples, 
            #"ERROR": error_msg,
            "CODE": state_node.total_code,
            #"PERCENTAGE": highest_value*100,
            #"BEST_CODE_ROLLOUT": best_code_rollout
        }
        
        #print("subs:",subs)
        #print("\n\nself.g_base_prompt\n",self.g_base_prompt)
        
        prompt = copy.deepcopy(self.g_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in self.g_base_prompt:
                    self.logger.warning(f"Key {key} not found in prompt file.")
                else:
                    prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
    
    def form_fix_prompt(self, state_node, highest_value, best_code_rollout):
        if state_node.extra_info is not None and "exception" in state_node.extra_info:
            error_msg = f"The code you wrote raised an exception: {state_node.extra_info['exception']}. Try fixing it or avoiding it for this next completion."
        else:
            error_msg = ""

        subs = {
            #"EXAMPLES": examples, 
            "ERROR": error_msg,
            "CODE": state_node.full_code,
            #"PERCENTAGE": highest_value*100,
            #"BEST_CODE_ROLLOUT": best_code_rollout
        }
        
        #print("subs:",subs)
        #print("\n\nself.base_prompt\n",self.base_prompt)
        
        prompt = copy.deepcopy(self.f_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in self.f_base_prompt:
                    raise ValueError(f"Key {key} not found in prompt file.")
                prompt = prompt.replace(f"{{{key}}}", str(value))
                #prompt = self.base_prompt.replace(f"{{key}}", str(value)) # buggy line
            
        return prompt
    
    def form_1_step_improve_prompt(self, state_node):
        incorrect_code = state_node.full_code
        misclassified_transition = state_node.extra_info['misclassified_transition']
        
        rtfm_format = self.env_name == 'rtfm'
        transition_inputs = get_and_format_transition_input(misclassified_transition, rtfm_format)
        ground_truth_transition = get_and_format_gt_transition(
            misclassified_transition, rtfm_format, self.check_valid_actions)
        predicted_transition = get_and_format_pred_transition(
            misclassified_transition, rtfm_format, self.check_valid_actions)

        subs = {
            "CODE":incorrect_code,
            "TRANSITION_INPUTS":transition_inputs,
            "GT_PREDICTION":ground_truth_transition,
            "PREDICTION":predicted_transition
        }

        prompt = copy.deepcopy(self.i_base_prompt)
        if subs is not None:
            for key, value in subs.items():
                if key not in prompt:
                    raise ValueError(f"Key {key} not found in prompt file.")
                prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt
    
    def check_code_world_model(self, code_env):
        valid, all_valid, success_rates, success_rate, all_misclassified_transitions = check_code_world_model(
            code_env, self.transitions, check_valid_actions=self.check_valid_actions, continuous_state=self.continuous_state
        )
        
        if len(all_misclassified_transitions) > 0:
            first_misclassified_transition = all_misclassified_transitions[0]
        else:
            first_misclassified_transition = None

        return success_rate, first_misclassified_transition
    
    def step(self, state_node, action_type, highest_value, best_code_rollout):
        prompt = self.form_prompt(state_node, action_type, highest_value, best_code_rollout) # ok
        self.logger.info("Prompt: \n", prompt)
        # Here we expect only the new part of the code and not the one included in the prompt
        # This is expected to be until the end of the program
        extra_info = {}
        completion = self.LLM.get_completion(prompt, exclude_prompt=True)
        
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
        extra_info = {}
        try:
            self.logger.info("Full program: \n", full_program)
            full_program = autopep8.fix_code(full_program)
            path = os.path.join(PROJECT_ROOT, "src", "gen_code_world_model.py")
            self.logger.info("Writing code to file: ", path)
            with open(path, 'w') as f:
                f.write(full_program)

            import src.gen_code_world_model as gen_code_world_model
            gen_code_world_model = reload(gen_code_world_model)
            # Remove generated code file
            self.logger.info("Removing tmp file: ", path)
            os.remove(path)
            # local_namespace = {}
            # exec(full_program, globals(), local_namespace)
            # Environment = local_namespace['Environment']  # Access the Environment class from the local namespace
    
            code_env = gen_code_world_model.Environment()
            prediction_success_rate, first_misclassified_transition = self.check_code_world_model(code_env)
            extra_info['bug'] = False
        except Exception as e:
            exception_info = traceback.format_exc()
            self.logger.info("An error occurred:\n", exception_info)
            exception_info, line_number = code_helpers.extract_relevant_traceback(exception_info) # clean error message
            if line_number:
                self.logger.info(f"Error on line {line_number}")
                tot_lines = len(full_program.split('\n'))
                if line_number-1 > tot_lines:
                    self.logger.warning(f"Line number {line_number} is greater than the number of lines in the program ({tot_lines})")
                else: 
                    line = full_program.split("\n")[line_number-1].strip()
                    if line and line != '':
                        self.logger.info(f"Line {line_number}: {line}")
                        # TODO keep an eye on this, it seems like the line is not always accurate
                        exception_info += f"\nThe line of code that caused the error is:\n`{line}`"
            
            self.logger.info("\nParsed error:\n", exception_info)
            extra_info['exception'] = exception_info
            prediction_success_rate = 0
            first_misclassified_transition = None
            #state_node.value_sum += 0.99 # This is set very high to give priority to fix actions on bugged nodes, but avoids getting backpropagated
            extra_info['bug'] = True
            
        extra_info['misclassified_transition'] = first_misclassified_transition
        return full_program, prediction_success_rate, extra_info