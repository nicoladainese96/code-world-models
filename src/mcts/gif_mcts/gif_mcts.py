import os
import sys
import numpy as np

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.DEBUG)

from . import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.mcts.mcts_visualization import print_tree
from src.code_helpers import save_tree, create_directory

def save_results(save_path, best_code_rollout, highest_value, extra_info):
    """
    Save results of the GIF-MCTS for code generation to file.
    """
    tree_out_dir = os.path.join(save_path, "tree")
    create_directory(tree_out_dir)
    
    with open(os.path.join(save_path, 'best_code_rollout.py'), 'w') as f:
        f.write(best_code_rollout)

    root = extra_info['root']
    mcts_state = {
        'best_code_rollout': best_code_rollout,
        'highest_value': highest_value,
        'extra_info': extra_info,
    }

    save_tree(tree_out_dir, root, prefix='', save_python_files=False, mcts_state=mcts_state)

class GIF_MCTS():
    def __init__(self,
                 simulator,
                 node_length,
                 ucb_c,
                 discount,
                 max_actions,
                 eps,
                 v_g_new_init,
                 g_counts_init,
                 v_f_new_init,
                 f_counts_init,
                 v_i_new_init,
                 i_counts_init,
                 max_fix_chain_length=3,
                 logger=DEFAULT_LOGGER,
                 save_path=None,
                 allow_generate=True,
                 allow_improve=True,
                ):
        """
        Monte Carlo Tree Search assuming determinstic dynamics with a Surrogate simulator of the
        coding process with an LLM.
        
        simulator: object
            Instance of [class to be defined]
        ucb_c:
            Constantused in the UCB1 formula for trees
            UCB(s,a) = Q(s,a) + ucb_c*sqrt(log N(s,a)/(\sum_b N(s,b)))
        discount:
            discoung factor gamma of the MDP.
        max_actions: int
            Number of actions to be taken at most from the the leaf node to the end of a rollout.
          v_new: float in (-1,1)
            What is default value for unexplored actions in the p-UCT formula. Higher is more optimistic and 
            increases exploration, but decreases the influence of the prior in the first few decisions.
            Suggested values: 0, 0.5 or 1.
        """
        self.simulator = simulator
        self.Vg_predictor = ValuePredictor(c=g_counts_init)
        self.Vf_predictor = ValuePredictor(c=f_counts_init)
        self.Vi_predictor = ValuePredictor(c=i_counts_init)
        self.node_length = node_length
        self.n_nodes = 0
        self.ucb_c = ucb_c
        self.discount = discount
        self.max_actions = max_actions
        self.eps = eps
        self.max_fix_chain_length = max_fix_chain_length
        
        # Generate an array of evenly spaced values for the bug value
        # The index represents the number of fixes left, so these go from 0 (no fixes left) to 0.99 (max_fix_chain_length fixes left)
        self.bug_values = np.linspace(0.0, 0.99, max_fix_chain_length+1) if max_fix_chain_length > 0 else [0.0]
        
        # Init prior values for generate and fix actions
        self.v_g_sum = v_g_new_init*g_counts_init
        self.g_count = g_counts_init
        self.v_g_stats = []
        
        # not particularly useful but let's see
        self.v_f_sum = v_f_new_init*f_counts_init
        self.f_count = f_counts_init
        self.v_f_stats = []
        
        self.v_i_sum = v_i_new_init*i_counts_init
        self.i_count = i_counts_init
        self.v_i_stats = []
        
        self.max_value = 1.0
        
        self.allow_generate = allow_generate # Used to disable the generate action for ablation
        self.allow_improve = allow_improve # Used to disable the improve action for ablation

        self.logger = logger
        self.logger.info("Allow generate: ", self.allow_generate)
        self.logger.info("Allow improve: ", self.allow_improve)
        self.save_path = save_path
        create_directory(self.save_path)
        
    def run(self, num_simulations, verbose=True):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary.
        
        num_simulations: int
            Number of simulations to run
      
        """
        self.root = CodeStateNode(logger=self.logger, allow_generate=self.allow_generate, allow_improve=self.allow_improve)
        self.root.initial_actions = ['g1'] # explicitly mask out the feedback action 'f1' at the root node (nothing to refine)
        self.n_nodes = 0
        self.root.expand(
            node_code = '', 
            reward = 0, 
            done = 0, 
            node_length = self.node_length, # this is wrong, but let's go with it for the time being
            ancestors_length = 0,
            ancestors_code = '',
            full_code = '',
            extra_info={'bug':False, 'fixes_left':self.max_fix_chain_length},
            node_id=self.n_nodes,
            action_type='root'
        )
        self.n_nodes += 1
        self.root.visit_count += 1 
        root = self.root   
        
        self.logger.debug('v_g_sum', self.v_g_sum)
        self.logger.debug('g_count', self.g_count)
        self.logger.debug('v_f_sum', self.v_f_sum)
        self.logger.debug('f_count', self.f_count)
        self.logger.debug('v_i_sum', self.v_i_sum)
        self.logger.debug('i_count', self.i_count)
            
        highest_value = 0
        value_list = []
        best_code_rollout = None
        
        max_tree_depth = 0    
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            self.logger.info("Simulation %d started."%(n+1))
            state_node = root
            search_path = [state_node]
            current_tree_depth = 0
            
            ### Selection phase until leaf node is reached ###
            new_transition = False # leaf node iff state_node is terminal or reached through new transition
            while not new_transition:
                current_tree_depth += 1
                # Select
                action, action_type, next_state_node, new_transition = self.select(state_node)
                if new_transition:
                    self.logger.info('n =', n, 'selected new action', action, 'at depth', current_tree_depth)
                else:
                    self.logger.info('n =', n, 'selected a =', action, 'at depth', current_tree_depth)
                
                # print("Current tree depth: ", current_tree_depth)
                # print("Action selected: ", action)
                # print("New transition: ", new_transition)
                # print("Child node terminal: ", next_state_node.terminal)
                
                if next_state_node.terminal or new_transition:
                    # Exit from the search when a terminal node or a new node is encountered
                    break
                else:
                    search_path.append(next_state_node) 
                    state_node = next_state_node

            ### Expansion and value prediction of the leaf node ###
            self.logger.info("Expansion and value simulation phase started")
            
            # Expand and simulate value
            expanded_next_state_node, value, value_plus_reward, code_rollout = self.expand_and_simulate_value(next_state_node, state_node, action_type, highest_value, best_code_rollout)
            search_path.append(expanded_next_state_node) 
            value_list.append(value_plus_reward)
            
            # Update value predictor
            self.update_value_predictor(value_plus_reward, state_node, action_type)

            # Dynamically update the global value statistics
            if action_type == 'generate':
                self.v_g_sum += value_plus_reward
                self.g_count += 1
                self.logger.debug('v_g_sum', self.v_g_sum)
                self.logger.debug('g_count', self.g_count)
                self.logger.debug('self.Vg_predictor.n', self.Vg_predictor.n)
                self.logger.debug('self.Vg_predictor.lr', self.Vg_predictor.lr)
            elif action_type == 'fix':
                self.v_f_sum += value_plus_reward
                self.f_count += 1
                self.logger.debug('v_f_sum', self.v_f_sum)
                self.logger.debug('f_count', self.f_count)
                self.logger.debug('self.Vf_predictor.n', self.Vf_predictor.n)
                self.logger.debug('self.Vf_predictor.lr', self.Vf_predictor.lr)
            else:
                self.v_i_sum += value_plus_reward
                self.i_count += 1
                self.logger.debug('v_i_sum', self.v_i_sum)
                self.logger.debug('i_count', self.i_count)
                self.logger.debug('self.Vi_predictor.n', self.Vi_predictor.n)
                self.logger.debug('self.Vi_predictor.lr', self.Vi_predictor.lr)
            
            if value_plus_reward > highest_value or best_code_rollout is None:
                highest_value = value_plus_reward
                best_code_rollout = code_rollout
                
            if highest_value == self.max_value:
                self.logger.info("Correct solution found!")
                self.logger.info('Value: ', value)
                max_tree_depth = max(max_tree_depth, current_tree_depth)
                extra_info = {
                    "max_tree_depth": max_tree_depth,
                    "value_list": value_list,
                    "root":self.root,
                    "v_g_stats":self.v_g_stats,
                    "v_f_stats":self.v_f_stats,
                    "v_i_stats":self.v_i_stats
                }
                return best_code_rollout, highest_value, extra_info
                
            ### Backpropagation of the leaf node value along the seach_path ###
            self.logger.info("Backpropagation phase started")
            
            if expanded_next_state_node.bug:
                self.logger.info("Not backpropagating value because of bug")
                self.backprop(search_path, value, visit_only=True)
            else:
                self.backprop(search_path, value)
            
            self.check_for_bugs(state_node, expanded_next_state_node)
            # Tried this, but the visit statistics did not get normalized correctly (node value ended up > 1)
            
            max_tree_depth = max(max_tree_depth, current_tree_depth)
            print_tree({'root': self.root}, logger=self.logger, print_non_expanded=False)
            # print("Simulation %d done."%(n+1))
            
            # Backup to file
            if self.save_path is not None:
                extra_info = {
                    "max_tree_depth": max_tree_depth,
                    "value_list": value_list,
                    "root":self.root,
                    "v_g_stats":self.v_g_stats,
                    "v_f_stats":self.v_f_stats,
                    "v_i_stats":self.v_i_stats
                }
                self.logger.info("Saving tree to file")
                save_results(self.save_path, best_code_rollout, highest_value, extra_info)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "value_list": value_list,
            "root":self.root,
            "v_g_stats":self.v_g_stats,
            "v_f_stats":self.v_f_stats,
            "v_i_stats":self.v_i_stats
        }
        
        return best_code_rollout, highest_value, extra_info
    
    def check_for_bugs(self, parent, child):
        """
        Given a parent and a child node, check whether the parent is bugged. 
        
        If the parent is bugged and the child didn't solve the bug, set the 
        parent's value to 0 and make it a terminal node.
        
        If the child solved the bug, set the bug=False and the parent value equal to
        the child value.
        
        We assume that the child is expanded.
        """
        assert child.expanded == True, "Child node should be expanded!"
        
        if parent.bug:
            self.logger.info("Original parent value", parent.value())
            # if child.value()==0.0 or child.bug: # Child could potentially have solved the bug but still have value 0
            if child.bug and child.extra_info['fixes_left'] == 0:
                self.logger.warn(f'Child {child.node_id} did not solve bug with the available fixes, do not expand parent {parent.node_id} further')
                self.logger.debug(f'Parent action type: {parent.action_type}')
                # Traverse the tree upwards and set all parents to terminal until we reach a parent that wasn't a "fix" action
                while parent is not None and parent.action_type[0] == 'f':
                    self.logger.debug(f'Parent {parent.node_id} is a fix action, setting to terminal')
                    parent.value_sum = 0
                    parent.terminal = True
                    parent = parent.parent
                    self.logger.debug(f'Parent action type: {parent.action_type}')
                self.logger.debug(f'Parent {parent.node_id} is not a fix action, setting to terminal and ending propagation')
                parent.value_sum = 0
                parent.terminal = True
                
            elif child.bug:
                self.logger.info(f'Child {child.node_id} did not solve bug, but has fixes left, setting parent {parent.node_id} to child value {child.value()}')
                self.logger.debug(f'Parent action type: {parent.action_type}')
                # Traverse the tree upwards and set all parents value sum to the bug value until we reach a parent that wasn't a "fix" action
                while parent is not None and parent.action_type[0] == 'f':
                    self.logger.debug(f'Parent {parent.node_id} is a fix action, setting to bug value {self.bug_values[child.extra_info["fixes_left"]]}')
                    parent.value_sum = self.bug_values[child.extra_info['fixes_left']]*parent.visit_count
                    parent.bug = False
                    parent = parent.parent
                    self.logger.debug(f'Parent action type: {parent.action_type}')
                    
                self.logger.debug(f'Parent {parent.node_id} is not a fix action, setting to bug value {self.bug_values[child.extra_info["fixes_left"]]} and ending propagation')
                parent.value_sum = self.bug_values[child.extra_info['fixes_left']]*parent.visit_count
                parent.bug = False
                
            elif not child.bug:
                parent.bug = False
                parent.value_sum = child.value()*parent.visit_count
                self.logger.info(f'Child {child.node_id} solved bug of parent {parent.node_id} (new value: {child.value()})')
                while parent is not None and parent.action_type[0] == 'f':
                    self.logger.debug(f'Parent {parent.node_id} is a fix action, setting to child value {child.value()}')
                    parent.value_sum = child.value()*parent.visit_count
                    parent = parent.parent
                    self.logger.debug(f'Parent action type: {parent.action_type}')
                self.logger.debug(f'Parent {parent.node_id} is not a fix action, setting to child value {child.value()} and ending propagation')
                parent.value_sum = child.value()*parent.visit_count
        else:
            self.logger.info("Parent.bug", parent.bug)
            self.logger.info("Child value", child.value())
            self.logger.info("Child fixes left", child.extra_info['fixes_left'])
            self.logger.info("Final parent value", parent.value())
                
    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*np.sqrt(np.log(parent.visit_count)/(child.visit_count+self.eps))
        self.logger.debug('exploration_term', exploration_term)

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.value()
            self.logger.debug(f'Value of {parent.node_id} -> {child.action} = {value_term:.4f} + {exploration_term:.4f} = {value_term + exploration_term:.4f}')
        else:
            # Logic for new actions
            if 'g' in child.action:
                action_type = 'generate'
            elif 'f' in child.action:
                action_type = 'fix'
            else:
                action_type = 'improve'
            
            # Count the number of actions of the same type present in the parent
            n_same_type = sum([1 for a in parent.children.keys() if action_type[0] in a]) - 1
            self.logger.debug('n_same_type', n_same_type)
            exploration_term = self.ucb_c*np.sqrt(np.log(parent.visit_count)/(n_same_type+self.eps))
            self.logger.debug('new exploration_term', exploration_term)
                            
            v_L = self.get_local_value(parent, action_type) # can be v_g_L or v_f_L
            v_G = self.get_global_value(action_type) # can be v_g_G or v_f_G
            self.logger.debug('child.action',child.action)
            self.logger.debug('v_L: ', v_L)
            self.logger.debug('v_G: ', v_G)
            if action_type == 'generate':
                value_term = self.Vg_predictor.forward(v_G, v_L).item()
            elif action_type == 'fix':
                value_term = self.Vf_predictor.forward(v_G, v_L).item()
            else:
                value_term = self.Vi_predictor.forward(v_G, v_L).item()
                
            self.logger.debug('value_term: ', value_term)
            value_term = min(max(value_term, 0.0),1.0)
            self.logger.debug('value_term (in [0,1]): ', value_term)
            vl = f'{v_L:.4f}' if v_L is not None else 'None'
            vg = f'{v_G:.4f}' if v_G is not None else 'None'
            self.logger.info(f'Value of {parent.node_id} -> {child.action} = {value_term:.4f} + {exploration_term:.4f} = {value_term + exploration_term:.4f}, v_L: {vl}, v_G: {vg}')
            
        return value_term + exploration_term, value_term, exploration_term
    
    def update_value_predictor(self, value_plus_reward, state_node, action_type):
        v_L = self.get_local_value(state_node, action_type) # can be v_g_L or v_f_L
        v_G = self.get_global_value(action_type) # can be v_g_G or v_f_G
            
        if action_type == 'generate':
            v_pred, loss = self.Vg_predictor.update(v_G, v_L, value_plus_reward)
            self.v_g_stats.append((v_G, v_L, v_pred, value_plus_reward, loss))
        elif action_type == 'fix':
            v_pred, loss = self.Vf_predictor.update(v_G, v_L, value_plus_reward)
            self.v_f_stats.append((v_G, v_L, v_pred, value_plus_reward, loss))
        else:
            v_pred, loss = self.Vi_predictor.update(v_G, v_L, value_plus_reward)
            self.v_i_stats.append((v_G, v_L, v_pred, value_plus_reward, loss))
        
    def get_local_value(self, state_node, action_type):
        if action_type == 'generate':
            prefix = 'g'
        elif action_type == 'fix':
            prefix = 'f'
        else:
            prefix = 'i'
            
        # Compute v_L as the average value of visited 'generate/fix' child nodes
        v_L = 0
        n_L = 0
        # Loop over children
        for a in state_node.children.keys():
            # Filter for 'generate' children which have been visited (and thus expanded)
            if prefix in a and state_node.children[a].expanded:
                v_L += state_node.children[a].value() # accumulate value
                n_L += 1
                
        if n_L == 0:
            v_L = None # no entries
        else:
            v_L = v_L/n_L # average of generate/fix children's value

        return v_L

    def get_global_value(self, action_type):
        if action_type == 'generate':
            return self.v_g_sum/self.g_count
        elif action_type == 'fix':
            return self.v_f_sum/self.f_count
        else:
            return self.v_i_sum/self.i_count
    
    def select(self, state_node):
        """
        Select which action to take in the input node through the p-UCT formula.
        
        If the new state is already present in the list of children corresponding to 
        the transitions sampled in the past, then just select that node.
        
        Return:
         - action: selected action (str)
         - next_state_node: selected next state node (CodeStateNode instance)
         - new_transition: whether the next_state_node is new or has already been visited (bool)
        """
        ### Usual part to select which action to take ###
        actions = []
        ucb_values = []
        value_terms = []
        exploration_terms = []
        
        for action, child in state_node.children.items():
            U, V, E = self.ucb_score(state_node, child)
            
            actions.append(action)
            ucb_values.append(U)
            value_terms.append(V)
            exploration_terms.append(E)
            
        actions = np.array(actions, dtype=object)
        value_terms = np.array(value_terms)
        exploration_terms = np.array(exploration_terms)
        ucb_values = np.array(ucb_values)
        
        # Select best action (split ties by random sampling among best actions)
        max_U = ucb_values.max()
        mask = (ucb_values==max_U)
        best_actions = actions[mask]
        action = np.random.choice(best_actions)
        
        if 'g' in action:
            action_type = 'generate'
        elif 'f' in action:
            action_type = 'fix'
        else:
            action_type = 'improve'
        
        # Select corresponding next state node
        next_state_node = state_node.children[action]

        # For debugging, print everything if verbose
        self.logger.info("actions: ", actions)
        self.logger.info("value_terms: ", value_terms)
        self.logger.info("exploration_terms: ", exploration_terms)
        self.logger.info("ucb_values: ", ucb_values)
        self.logger.info("max_U: ", max_U)
        # self.logger.info("mask: ", mask)
        self.logger.info("best_actions: ", best_actions)
        self.logger.info("Action selected: ", action)
        self.logger.info("Action type: ", action_type)
        
        # Check if the next_state_node is a new node or has already been visited
        if next_state_node.visit_count != 0:
            new_transition = False # already visited
        else:
            new_transition = True # new node
            
            if not state_node.bug: # don't add extra actions to bugged nodes, only one single fix
                # If the action was a generate action, add a generate action, else, add a fix action
                # TODO don't add fix actions to force chain? Should be enforced by the not state_node.bug condition I think
                if action[0] == 'g' and not self.allow_generate:
                    # Avoid adding new generates at the root node
                    self.logger.info("Generate action not allowed, skipping")
                else:
                    new_action = f'{action[0]}{int(action[1:])+1}' # remove the 'g' or 'f' or 'i' in front, then cast to int and add 1
                    self.logger.info("New action added at the parent: ", new_action)
                    state_node.children[new_action] = CodeStateNode(logger=self.logger, allow_generate=self.allow_generate, allow_improve=self.allow_improve)
                    state_node.children[new_action].action = new_action

        return action, action_type, next_state_node, new_transition
    
    def expand_and_simulate_value(self, next_state_node, state_node, action_type, highest_value, best_code_rollout):

        first_new_l_lines, ancestors_code, full_program, new_length, ancestors_len, reward, done, value, extra_info = self.simulator.step(state_node, action_type, highest_value, best_code_rollout)

        # Expand the new_state_node
        self.logger.info("first_new_l_lines: \n", first_new_l_lines)
        self.logger.info("new_length: ", new_length)
        self.logger.info("reward: ", reward)
        self.logger.info("done: ", done)
        self.logger.info("value: ", value)
        
        # If child didn't fix the bug of the parent AND the action was a fix action, decrease the number of fixes left
        if extra_info['bug'] and action_type == 'fix':
            extra_info['fixes_left'] = state_node.extra_info['fixes_left'] - 1
        else:
            extra_info['fixes_left'] = self.max_fix_chain_length

        assert extra_info['fixes_left'] >= 0, "Fixes left should be >= 0"
        self.logger.debug('Fixes left:', extra_info['fixes_left'])
        
        if extra_info['bug']:
            self.logger.info("Bug found, fixes left: ", extra_info['fixes_left'], "bug value: ", self.bug_values[extra_info['fixes_left']])
        
        next_state_node.expand(
            first_new_l_lines, 
            reward, 
            done, 
            new_length, 
            ancestors_len,
            ancestors_code,
            full_program,
            action_type=action_type,
            extra_info=extra_info,
            node_id=self.n_nodes,
            bug_value=self.bug_values[extra_info['fixes_left']]
        )
        self.n_nodes += 1
        next_state_node.parent = state_node
        
        code_rollout = full_program
        next_state_node.full_code_value = value + reward
        self.logger.info('value + reward =', value + reward, 'bug:', extra_info['bug'])
        return next_state_node, value, value+reward, code_rollout
    
    def backprop(self, search_path, value, visit_only=False):
        """
        Update the value sum and visit count of all state nodes along the search path. 
        """
        # How the chain of values and Q-values works
        # Q_{T-2} = r_{T-2} + \gamma V_{T-1} -> V_{T-1} = Q_{T-1} -> Q_{T-1} = r_{T-1} + \gamma V_T -> v_T
        for node in reversed(search_path):
            node.value_sum += node.value() if visit_only else value
            node.visit_count += 1
            value = node.reward + self.discount*value # actually a Q-value