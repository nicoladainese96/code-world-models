import numpy as np
import copy
import torch
import torch.nn.functional as F
import hashlib

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

def hash_frame(frame):
    h = hashlib.sha256()
    h.update(str(frame).encode('utf-8'))
    return h.hexdigest()


class StochasticStateNode():
    def __init__(self, logger=DEFAULT_LOGGER):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.reward = 0
        self.simulator = None
        self.expanded = False
        self.terminal = False
        self.simulator_dict = None
        self.frame = None
        self.full_action_space = None # comprehends also moves that are not allowed in the node
        self.transition_id = None # hash
    
        self.logger = logger

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 
   
    def expand(self, frame, valid_actions, reward, done, simulator, full_action_space):
        self.expanded = True
        self.logger.debug("Valid actions as child: ", valid_actions)
        self.logger.debug("Terminal node: ", done)
        self.full_action_space = full_action_space
        self.frame = frame
        self.reward = reward
        self.terminal = done
        self.valid_actions = valid_actions
        if not done:
            for action in valid_actions:
                self.children[action] = StochasticStateActionNode()
                self.children[action].action = action
        self.simulator_dict = simulator.save_state_dict()
    
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
            Qs[action] = child.Q_value()
        return Qs
    
    def get_children_visit_counts(self):
        Ns = np.zeros(self.full_action_space) # check this
        for action, child in self.children.items():
            Ns[action] = child.visit_count
        return Ns
    
    def render(self, simulator):
        if self.simulator_dict is not None:
            simulator.load_state_dict(self.simulator_dict)
            simulator.render()
        else:
            raise Exception("Node simulator not initialized yet.")
            
    def get_simulator(self, simulator):
        if self.simulator_dict is not None:
            # load a deepcoy of the simulator_dict, so that the internal variable remains unchanged
            simulator.load_state_dict(copy.deepcopy(self.simulator_dict)) 
            return simulator
        else:
            self.logger.error("Trying to load simulator_dict, but it was never instantiated.")
            raise NotImplementedError()


class StochasticStateActionNode():
    def __init__(self, logger=DEFAULT_LOGGER):
        self.visit_count = 0
        self.Q_value_sum = 0
        self.children = {}
        self.action = None # has it ever been used?

        self.logger = logger
        
    def Q_value(self):
        if self.visit_count == 0:
            return 0
        return self.Q_value_sum / self.visit_count 

class StochasticMCTSNovelty():
    def __init__(self,
                 root_frame,
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 max_actions,
                 root=None,
                 render=False,
                 logger=DEFAULT_LOGGER,
                 use_novelty_bonus=True,
                 distance_for_novelty=0.001,
                 novelty_bonus=0.1
                 
                ):
        """
        Monte Carlo Tree Search assuming stochastic dynamics.
        
        root_frame: dict of torch tensors
            Processed frame of the current state (a.k.a. root node).
        simulator: object
            Instance of TrueSimulatorWithGroundTruthClasses from ./env_utils .
        valid_actions: list
            List of valid actions for the root node.
        ucb_c:
            Constantused in the UCB1 formula for trees
            UCB(s,a) = Q(s,a) + ucb_c*sqrt(log N(s,a)/(\sum_b N(s,b)))
        discount:
            discoung factor gamma of the MDP.
        max_actions: int
            Number of actions to be taken at most from the root node to the end of a rollout.
        root: object
            The child of an old root node; use it to keep all the cached computations 
            from previous searches with a different root node. (optional)
        """
        self.simulator = simulator
        self.original_dict = simulator.save_state_dict()
        self.valid_actions = valid_actions
        self.full_action_space = simulator.action_space
        self.ucb_c = ucb_c
        self.discount = discount
        self.root_frame = root_frame
        self.root = root 
        self.render = render
        self.max_actions = max_actions
        
        self.list_of_frames = []
        self.use_novelty_bonus = use_novelty_bonus
        self.distance_for_novelty = distance_for_novelty
        self.novelty_bonus = novelty_bonus
        
        self.logger = logger
        
    def check_frame_is_novel(self, frame):
        
        def distance(f1, f2):
            distance = np.linalg.norm(f1-f2)/np.linalg.norm(f1) 
            #self.logger.info("distance", distance)
            return distance
        
        # Should be different enough from every single frame seen so far
        novel = np.all([distance(frame, f)>self.distance_for_novelty for f in self.list_of_frames])
        #self.logger.info("novel", novel)
        
        # Now consider the frame as "seen"
        self.list_of_frames.append(frame)
        
        return novel
    
    def get_novelty_bonus(self, frame):
        novel = self.check_frame_is_novel(frame)
        if novel:
            return self.novelty_bonus
        else:
            return 0.0
        
    def get_subtree(self, action, new_frame):
        """
        Returns the subtree whose root node is the current root's child corresponding to
        the given action and transition id (since it's stochastic). 
        If nothing is found, then return None.
        """
        raise NotImplementedError
        
        # TO BE CHECKED
        #state_action_node = self.root.children[action]
        #env_transition_id = hash(str(new_frame['name'].cpu().numpy()))
        #transition_ids = np.array(list(state_action_node.children.keys()))
        # if the transition actually occurred in the environment has already been encountered
        # in the tree search, select the subtree which has that state node as root
        #if np.any(env_transition_id==transition_ids):
        #    new_root = state_action_node.children[env_transition_id]
        #else:
        #    new_root = None
            
        #return new_root
    
    def run(self, num_simulations, default_Q=1.0):
        """
        Runs num_simulations searches starting from the root node corresponding to the internal
        state of the simulator given during initialization.
        Returns the root node and an extra_info dictionary.
        
        num_simulations: int
            Number of simulations to run
        default_Q: float in (-1,1)
            What is default value for unexplored actions in the p-UCT formula. Higher is more optimistic and 
            increases exploration, but decreases the influence of the prior in the first few decisions.
            Suggested values: 0, 0.5 or 1.
        """
        self.default_Q = default_Q
        
        if self.root is None or self.root.visit_count==0:
            self.root = StochasticStateNode() 
                
            self.root.expand(
                self.root_frame,
                self.valid_actions,
                0, # reward to get to root
                False, # terminal node
                self.simulator, # state of the simulator at the root node 
                self.full_action_space
                )
                
            self.root.visit_count += 1 
                
        max_tree_depth = 0 # keep track of the maximum depth of the tree (hard to compute at the end) 
        root = self.root
           
        for n in range(num_simulations):
            ### Start of a simulation/search ###
            self.logger.debug("Simulation %d started."%(n+1))
            state_node = root
            # make sure that the simulator internal state is reset to the original one
            self.simulator.load_state_dict(root.simulator_dict)
            search_path = [state_node]
            current_tree_depth = 0
            if self.render:
                state_node.render(self.simulator)
            ### Selection phase until leaf node is reached ###
            new_transition = False # leaf node iff state_node is terminal or reached through new transition
            while not new_transition:
                current_tree_depth += 1
                action, state_action_node, state_node, new_transition = self.select_and_expand(state_node)
                if self.render:
                    state_node.render(self.simulator)
                self.logger.debug("Current tree depth: ", current_tree_depth)
                self.logger.debug("Action selected: ", action)
                self.logger.debug("New transition: ", new_transition)
                self.logger.debug("Child node terminal: ", state_node.terminal)
                # always append both nodes
                search_path.append(state_action_node) 
                search_path.append(state_node) 
                
                if state_node.terminal or new_transition:
                    # Exit from the search when a terminal node or a new node is encountered
                    break

            ### Value prediction of the leaf node ###
            self.logger.debug("Value simulation phase started")
            value = self.simulate_value(state_node)
            
            ### Backpropagation of the leaf node value along the seach_path ###
            self.logger.debug("Backpropagation phase started")
            self.backprop(search_path, value)
        
            max_tree_depth = max(max_tree_depth, current_tree_depth)
            self.logger.debug("Simulation %d done."%(n+1))
            
        extra_info = {
            "max_tree_depth": max_tree_depth
        }
        # just a check to see if root works as a shallow copy of self.root
        assert root.visit_count == self.root.visit_count, "self.root not updated during search"
        
        # make sure that the simulator internal state is reset to the original one
        self.simulator.load_state_dict(root.simulator_dict)
        return root, extra_info
                
    def select_and_expand(self, state_node):
        """
        Select which action to take in the input node through the p-UCT formula.
        Sample a transition for that (state, action) pair (i.e. get a StochasticStateNode
        that is going to be a child of the StochasticStateActionNode) in the simulator.
        If the new state is already present in the list of children corresponding to the transitions 
        sampled in the past, then just select that node, otherwise initialize a StochasticStateNode,
        add it to the list of children and expand it.
        Return both the StochasticStateActionNode and the StochasticStateNode.
        """
        ### Usual part to select which action to take ###
        actions = []
        ucb_values = []
        value_terms = []
        exploration_terms = []
        for action, child in state_node.children.items():
            actions.append(action)
            U, V, E = self.ucb_score(state_node, child)
            ucb_values.append(U)
            value_terms.append(V)
            exploration_terms.append(E)
        actions = np.array(actions)
        self.logger.debug("actions: ", actions)
        
        value_terms = np.array(value_terms)
        self.logger.debug("value_terms: ", value_terms)
        
        exploration_terms = np.array(exploration_terms)
        self.logger.debug("exploration_terms: ", exploration_terms)
        
        ucb_values = np.array(ucb_values)
        self.logger.debug("ucb_values: ", ucb_values)
        
        max_U = ucb_values.max()
        self.logger.debug("max_U: ", max_U)
        
        mask = (ucb_values==max_U)
        self.logger.debug("mask: ", mask)
        
        best_actions = actions[mask]
        self.logger.debug("best_actions: ", best_actions)
        
        action = np.random.choice(best_actions)
        self.logger.debug("Action selected: ", action)
        
        state_action_node = state_node.children[action]
        
        ### New part for stochastic MCTS ###
        simulator = state_node.get_simulator(self.simulator) # get a deepcopy of the simulator with the parent's state stored
        frame, valid_actions, reward, done, _ = self.simulator_step(simulator, action, state_node.frame) # this also updates the simulator's internal state
        
        # hash the frame 
        transition_id = hash_frame(frame)
        
        # check if the transition has already been sampled in the past
        transition_ids = np.array(list(state_action_node.children.keys()))
        
        if np.any(transition_id==transition_ids):
            # if that is the case, just select the new_state_node and return 
            new_transition = False
            new_state_node = state_action_node.children[transition_id]
        
        else:
            new_transition = True
            
            # init the new_state_node
            new_state_node = StochasticStateNode()
            
            # add transition to the children of the state_action_node
            state_action_node.children[transition_id] = new_state_node
            
            # expand the new_state_node
            self.logger.debug("valid_actions: ", valid_actions)
            self.logger.debug("reward: ", reward)
            self.logger.debug("done: ", done)
            
            # Add exploration bonus if applicable
            if self.use_novelty_bonus:
                novelty_bonus = self.get_novelty_bonus(frame)
                reward += novelty_bonus
                #self.logger.info("novelty_bonus: ", novelty_bonus)
                #self.logger.info("reward: ", reward)
                
            new_state_node.expand(frame, valid_actions, reward, done, simulator, self.full_action_space)
        return action, state_action_node, new_state_node, new_transition
    
    def simulator_step(self, simulator, action, frame):
        return simulator.step(action)
    
    def ucb_score(self, parent, child, eps=1e-3):
        """
        The score for a node is based on its value, plus an exploration bonus.
        """
        exploration_term = self.ucb_c*np.sqrt(np.log(parent.visit_count)/(child.visit_count+eps))

        if child.visit_count > 0:
            # Mean value Q
            value_term = child.Q_value()
        else:
            value_term = self.default_Q # just trying
            
        return value_term + exploration_term, value_term, exploration_term
    
    def simulate_value(self, state_node):
        """
        Simulate a rollout with a random policy starting from the input node
        until the end of the episode or self.max_actions are reached 
        (also considering the current depth of the input node from the root)
        """
        if not state_node.terminal:
            simulator = state_node.get_simulator(self.simulator)
            valid_actions = state_node.valid_actions
            steps = self.max_actions
            cum_discounted_reward = 0
            for i in range(steps):
                action = np.random.choice(valid_actions)
                frame, valid_actions, reward, done, _  = self.simulator_step(simulator, action, state_node.frame)
                
                # Add exploration bonus if applicable
                if self.use_novelty_bonus:
                    novelty_bonus = self.get_novelty_bonus(frame)
                    reward += novelty_bonus
                    #self.logger.info("novelty_bonus: ", novelty_bonus)
                    #self.logger.info("reward: ", reward)
                    
                cum_discounted_reward += (self.discount**i)*reward
                if done:
                    break
        else:
            cum_discounted_reward = 0
        return cum_discounted_reward
    
    def backprop(self, search_path, value):
        """
        Update the value sum and visit count of all state nodes along the search path. 
        Does the same but updating the Q values for the state action nodes.
        search_path starts with a StochasticStateNode and also ends with one, in the middle the two classes are alternated.
        """
        # How the chain of values and Q-values works
        # Q_{T-2} = r_{T-2} + \gamma V_{T-1} -> V_{T-1} = Q_{T-1} -> Q_{T-1} = r_{T-1} + \gamma V_T -> v_T
        for node in reversed(search_path):
            if isinstance(node, StochasticStateNode):
                node.value_sum += value
                node.visit_count += 1
                value = node.reward + self.discount*value # actually a Q-value
            else:
                node.Q_value_sum += value
                node.visit_count +=1
                value = value # just to show that the value flows unchanged through the state-action
