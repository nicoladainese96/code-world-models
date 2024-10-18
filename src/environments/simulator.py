import copy

class Simulator():
    """
    Generic class to wrap an Environment class in a Simulator, which can
    be used inside the (stochastic) MCTS class for planning. 
    """
    def __init__(self, env, action_space):
        """
        env: instance of Environment class
        action_space: int, total number of possible actions 
        """
        self.env = env
        self.action_space = action_space
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame, valid_actions, extra_info = self.env.reset()
        return frame, valid_actions, extra_info
    
    def step(self, action, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        frame, valid_actions, reward, done, extra_info = self.env.step(action, *args, **kwargs)
        return frame, valid_actions, reward, done, extra_info
    
    def render(self):
        raise NotImplementedError
        
    def save_state_dict(self):
        """
        Saves all internal variables of the env instance in a dictionary with deepcopy.
        Can be used to take a snapshot of the env, to restore it back to that state
        later on.
        """
        state_dict = {}
        for k in self.env.__dict__.keys():
            if k != 'configs':
                state_dict[k] = copy.deepcopy(self.env.__dict__[k])
        return state_dict
        
    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the env to the one of the state_dict.
        """
        for k in state_dict.keys():
            setattr(self.env, k, state_dict[k])
            
            
class GymSimulator():
    """
    Generic class to wrap an Environment class in a Simulator, which can
    be used inside the (stochastic) MCTS class for planning. 
    """
    def __init__(self, env):
        """
        env: instance of Environment class
        """
        self.env = env
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame, extra_info = self.env.reset()
        return frame, extra_info
    
    def step(self, action, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        frame, reward, done, terminated, extra_info = self.env.step(action, *args, **kwargs)
        return frame, reward, done, terminated, extra_info
    
    def render(self):
        raise NotImplementedError
        
    def save_state_dict(self):
        """
        Saves all internal variables of the env instance in a dictionary with deepcopy.
        Can be used to take a snapshot of the env, to restore it back to that state
        later on.
        """
        state_dict = {}
        for k in self.env.__dict__.keys():
            if k != 'configs':
                try:
                    state_dict[k] = copy.deepcopy(self.env.__dict__[k])
                except Exception as e:
                    #print(e)
                    pass
                
        return state_dict
        
    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the env to the one of the state_dict.
        """
        for k in state_dict.keys():
            setattr(self.env, k, state_dict[k])
            
class GymSimulatorValidActions():
    """
    Generic class to wrap an Environment class in a Simulator, which can
    be used inside the (stochastic) MCTS class for planning. 
    """
    def __init__(self, env, valid_actions):
        self.env = env
        self.valid_actions = valid_actions
        self.action_space = len(valid_actions)
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame, extra_info = self.env.reset()
        return frame, self.valid_actions
    
    def step(self, action, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        frame, reward, done, terminated, extra_info = self.env.step(action, *args, **kwargs)
        return frame, self.valid_actions, reward, done, extra_info
    
    def render(self):
        raise NotImplementedError
        
    def save_state_dict(self):
        """
        Saves all internal variables of the env instance in a dictionary with deepcopy.
        Can be used to take a snapshot of the env, to restore it back to that state
        later on.
        """
        state_dict = {}
        for k in self.env.__dict__.keys():
            if k != 'configs':
                try:
                    state_dict[k] = copy.deepcopy(self.env.__dict__[k])
                except Exception as e:
                    #print(e)
                    pass
                
        return state_dict
        
    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the env to the one of the state_dict.
        """
        for k in state_dict.keys():
            setattr(self.env, k, state_dict[k])
        
            
class Connect4Simulator(Simulator):
    def __init__(self, env, action_space=7):
        self.env = env
        self.action_space = action_space
        
    def render(self):
        """
        Assuming env has an attribute x with the values of the board. 
        A better way would be to have a 'get_state' method built in in the env
        to return the state of the board, regardless of how the variable storing 
        it is called.
        """
        symbols = {0: ' ', 1: 'X', 2: 'O'}

        for row in range(self.env.x.shape[0]):
            print("|", end="")
            for col in range(self.env.x.shape[1]):
                print(f" {symbols[self.env.x[row, col]]} |", end="")
            print("\n+" + "---+" * self.env.x.shape[1])

        # Print column numbers at the bottom
        print(" ".join([f"  {i + 1}" for i in range(self.env.x.shape[1])]))
        
