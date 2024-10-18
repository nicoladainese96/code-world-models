import os
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

# Constants
L = 100
dp_dl = 1.0/100 # distribution of probability of sampling an error in 1 line of code
EDGE_ERROR_FRAC = 0.1 # fraction of errors made because of an edge error
CORE_ERROR_FRAC = 0.9  # fraction of errors made because of a core error

# adapted from src/mcts.py

class SurrogateStateNode():
    def __init__(self, logger=DEFAULT_LOGGER):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.expanded = False
        self.terminal = False
        self.full_action_space = None
        
        self.core_fault = None
        self.edge_fault = None
        self.ancestors_core_faults = None
        self.ancestors_edge_faults = None
        self.tot_core_faults = None
        self.tot_edge_faults = None
        
        self.node_length = None
        self.ancestors_length = None
        self.tot_length = None

        self.logger = logger
        
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 
    
    def expand(
        self, 
        faults, 
        reward, 
        done, 
        full_action_space, 
        node_length, 
        ancestors_length,
        ancestors_core_faults,
        ancestors_edge_faults
    ):
        self.expanded = True
        self.logger.info("Terminal node: ", done)
        self.full_action_space = full_action_space
        self.core_fault = faults[0]
        self.edge_fault = faults[1]
        self.ancestors_core_faults = ancestors_core_faults
        self.ancestors_edge_faults = ancestors_edge_faults
        self.tot_core_faults = ancestors_core_faults + 1 if faults[0] else ancestors_core_faults
        self.tot_edge_faults = ancestors_edge_faults + 1 if faults[1] else ancestors_edge_faults
        
        if done:
            # might not be the right formula, but does the job
            self.reward = ((1-CORE_ERROR_FRAC)**self.tot_core_faults)*\
            ((1-EDGE_ERROR_FRAC)**self.tot_edge_faults)*reward
            self.logger.debug('reward: ', reward)
            self.logger.debug('self.reward',self.reward)
        else:
            self.reward = reward
            
        self.terminal = done
        self.node_length = node_length
        self.ancestors_length = ancestors_length
        self.tot_length = node_length + ancestors_length
        
        if not done:
            for action in np.arange(full_action_space):
                self.children[action] = SurrogateStateNode()
                self.children[action].action = action
    
    def softmax_Q(self, T):
        Qs = self.get_Q_values()
        if T > 0:
            probs = F.softmax(Qs/T, dim=0)
        elif T==0:
            probs = torch.zeros(self.full_action_space) 
            a = torch.argmax(Qs)
            probs[a] = 1.
            
        sampled_action = torch.multinomial(probs, 1).item()
        return sampled_action, probs.cpu().numpy()
    
    def get_Q_values(self):
        Qs = -torch.ones(self.full_action_space)*np.inf
        for action, child in self.children.items():
            Qs[action] = child.value()
        return Qs
    
    def get_children_visit_counts(self):
        Ns = np.zeros(self.full_action_space) # check this
        for action, child in self.children.items():
            Ns[action] = child.visit_count
        return Ns
    
class SurrogateSimulator():
    def __init__(self, L, dp_dl, logger=DEFAULT_LOGGER):
        self.L = L
        self.dp_dl = dp_dl

        self.logger = logger
        
    def step(self, state_node, action):
        if state_node.node_length + state_node.tot_length < self.L:
            # Not terminal
            done = False
            reward = 0
            length = state_node.node_length # generate same length
        else:
            # Terminal
            done = True
            reward = 1 # to be suppressed by the errors in the code
            length = self.L - state_node.tot_length
        
        dp = self.dp_dl*length
        # With same prob dp, generate True or False variable for core faults and edge faults
        core_fault = np.random.choice([True,False], p=[dp,1.0-dp])
        edge_fault = np.random.choice([True,False], p=[dp,1.0-dp])
        faults = (core_fault, edge_fault)
        self.logger.debug('faults', faults)
        return length, faults, reward, done
    
# adapted from src/mcts.py
class SurrogateMCTS():
    def __init__(self,
                 simulator,
                 action_space,
                 node_length,
                 ucb_c,
                 discount,
                 max_actions,
                 eps,
                 v_new,
                 logger=DEFAULT_LOGGER
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
        self.full_action_space = action_space
        self.node_length = node_length
        self.ucb_c = ucb_c
        self.discount = discount
        self.max_actions = max_actions
        self.eps = eps
        self.v_new = v_new
        self.max_value = 1.0

        self.logger = logger
        
    def run(self, num_simulations):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary.
        
        num_simulations: int
            Number of simulations to run
      
        """
        self.root = SurrogateStateNode() 
        self.root.expand(
            faults = (False,False), 
            reward = 0, 
            done = 0, 
            full_action_space = self.full_action_space,
            node_length = self.node_length, # this is wrong, but let's go with it for the time being
            ancestors_length = 0,
            ancestors_core_faults = 0,
            ancestors_edge_faults = 0
            )
        self.root.visit_count += 1 
        root = self.root   
        
        highest_value = 0
        best_rollout = []
        
        max_tree_depth = 0    
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            self.logger.info("\nSimulation %d started."%(n+1))
            state_node = root
            search_path = [state_node]
            current_tree_depth = 0
            
            ### Selection phase until leaf node is reached ###
            new_transition = False # leaf node iff state_node is terminal or reached through new transition
            while not new_transition:
                current_tree_depth += 1
                action, state_node, new_transition = self.select_and_expand(state_node)

                self.logger.info("Current tree depth: ", current_tree_depth)
                self.logger.info("Action selected: ", action)
                self.logger.info("New transition: ", new_transition)
                self.logger.info("Child node terminal: ", state_node.terminal)
                
                # always append both nodes
                search_path.append(state_node) 
                
                if state_node.terminal or new_transition:
                    # Exit from the search when a terminal node or a new node is encountered
                    break

            ### Value prediction of the leaf node ###
            self.logger.info("Value simulation phase started")
            value, partial_rollout = self.simulate_value(state_node)
            
            if value > highest_value:
                highest_value = value
                best_rollout = [copy.deepcopy(node) for node in search_path+partial_rollout]
                
            if highest_value == self.max_value:
                self.logger.info("Correct solution found!")
                self.logger.info('Value: ', value)
                max_tree_depth = max(max_tree_depth, current_tree_depth)
                extra_info = {
                    "max_tree_depth": max_tree_depth
                }
                return best_rollout, highest_value, extra_info
                
            ### Backpropagation of the leaf node value along the seach_path ###
            self.logger.info("Backpropagation phase started")
            self.backprop(search_path, value)
        
            max_tree_depth = max(max_tree_depth, current_tree_depth)
            self.logger.info("Simulation %d done."%(n+1))
            
        extra_info = {
            "max_tree_depth": max_tree_depth
        }
        
        return best_rollout, highest_value, extra_info
                
    def select_and_expand(self, state_node):
        """
        Select which action to take in the input node through the p-UCT formula.
        
        If the child state node corresponding to the selected action is not expanded,
        sample the next state through the simulator and expand the node.
        If the new state is already present in the list of children corresponding to 
        the transitions sampled in the past, then just select that node.
        
        Return:
         - action: selected action (int)
         - next_state_node: selected next state node (SurrogateStateNode instance)
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
            
        actions = np.array(actions)
        value_terms = np.array(value_terms)
        exploration_terms = np.array(exploration_terms)
        ucb_values = np.array(ucb_values)
        
        # Select best action (split ties by random sampling among best actions)
        max_U = ucb_values.max()
        mask = (ucb_values==max_U)
        best_actions = actions[mask]
        action = np.random.choice(best_actions)
        
        # Select corresponding next state node
        next_state_node = state_node.children[action]

        # For debugging, print everything if verbose
        self.logger.debug("actions: ", actions)
        self.logger.debug("value_terms: ", value_terms)
        self.logger.debug("exploration_terms: ", exploration_terms)
        self.logger.debug("ucb_values: ", ucb_values)
        self.logger.debug("max_U: ", max_U)
        self.logger.debug("mask: ", mask)
        self.logger.debug("best_actions: ", best_actions)
        self.logger.debug("Action selected: ", action)
        
        # Check if the next_state_node is a new node or has already been visited
        if next_state_node.visit_count != 0:
            new_transition = False # already visited
        
        else:
            new_transition = True # new node
            
            # Use the simulator to get the properties of the next_state_node
            # Note: action is actually useless in our surrogate model
            # state_node is just used to keep track of the current program length
            length, faults, reward, done = self.simulator.step(state_node, action) 
            
            # Expand the new_state_node
            self.logger.debug("faults: ", faults)
            self.logger.debug("reward (undiscounted by faults): ", reward)
            self.logger.debug("done: ", done)
            next_state_node.expand(
                faults, 
                reward, 
                done, 
                self.full_action_space, 
                length, 
                state_node.tot_length,
                state_node.tot_core_faults,
                state_node.tot_edge_faults
            )
            vprint("reward (discounted by faults): ", next_state_node.reward)
        return action, next_state_node, new_transition
    
    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*np.sqrt(np.log(parent.visit_count)/(child.visit_count+self.eps))

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.value()
        else:
            value_term = self.v_new # Could be replaced by a dynamic value
            
        return value_term + exploration_term, value_term, exploration_term
    
    def simulate_value(self, state_node):
        """
        Simulate a rollout with a random policy starting from the input node
        until the end of the episode or self.max_actions are reached.
        """
        
        ### TODO: take into account all the faults in the final reward!
        partial_rollout = []
        if not state_node.terminal:
            steps = self.max_actions
            cum_discounted_reward = 0
            for i in range(steps):
                self.logger.debug('state_node.tot_core_faults', state_node.tot_core_faults)
                self.logger.debug('state_node.tot_edge_faults', state_node.tot_edge_faults)
                action = np.random.choice(np.arange(self.full_action_space))
                length, faults, reward, done = self.simulator.step(state_node, action) 
                next_state_node = SurrogateStateNode()
                next_state_node.expand(
                    faults, 
                    reward, 
                    done, 
                    self.full_action_space, 
                    length, 
                    state_node.tot_length,
                    state_node.tot_core_faults,
                    state_node.tot_edge_faults
                )
                
                cum_discounted_reward += (self.discount**i)*next_state_node.reward
                partial_rollout.append(next_state_node)
                if done:
                    break
                else:
                    state_node = next_state_node
        else:
            cum_discounted_reward = 0
        self.logger.debug("cum_discounted_reward", cum_discounted_reward)
        return cum_discounted_reward, partial_rollout
    
    def backprop(self, search_path, value):
        """
        Update the value sum and visit count of all state nodes along the search path. 
        """
        # How the chain of values and Q-values works
        # Q_{T-2} = r_{T-2} + \gamma V_{T-1} -> V_{T-1} = Q_{T-1} -> Q_{T-1} = r_{T-1} + \gamma V_T -> v_T
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.discount*value # actually a Q-value
           