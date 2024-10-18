import copy
import torch
import types
import numpy as np

class CWMSimulatorMCTS():
    def __init__(self, env, valid_actions):
        self.env = env
        self.valid_actions = valid_actions
        self.action_space = len(valid_actions)
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame = self.env.reset()
        return frame, self.valid_actions
    
    def step(self, action, frame, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        self.env.set_state(frame)
        frame, reward, done = self.env.step(action, *args, **kwargs)
        return frame, self.valid_actions, reward, done, None
    
    def render(self):
        raise NotImplementedError
        
    def save_state_dict(self):
        """
        Saves all internal variables of the env instance in a dictionary with deepcopy.
        Can be used to take a snapshot of the env, to restore it back to that state
        later on.
        """
        state_dict = {}
        for k, v in self.env.__dict__.items():
            if k != 'configs' and not isinstance(v, types.ModuleType):
                state_dict[k] = copy.deepcopy(v)
        return state_dict
        
    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the env to the one of the state_dict.
        """
        for k in state_dict.keys():
            setattr(self.env, k, state_dict[k])
            
class CWMSimulatorCEM():
    def __init__(self, env):
        self.env = env
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame = self.env.reset()
        extra_info = None
        return frame, extra_info
    
    def step(self, action, frame, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        self.env.set_state(frame)
        frame, reward, done = self.env.step(action, *args, **kwargs)
        terminated = done
        extra_info = None
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
        for k, v in self.env.__dict__.items():
            if k != 'configs' and not isinstance(v, types.ModuleType):
                state_dict[k] = copy.deepcopy(v)
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the env to the one of the state_dict.
        """
        for k in state_dict.keys():
            setattr(self.env, k, state_dict[k])
            
class StochasticCWMSimulator():
    def __init__(self, env, prior, device, action_space):
        self.env = env
        self.prior = prior.to(device)
        self.device = device
        self.action_space = action_space
        
    def reset(self):
        """
        Resets environment. Returns current state (frame) and possible valid_actions to take.
        """
        frame = self.env.reset()
        valid_actions = np.arange(self.action_space) # TODO: make this be returned from the env.reset method
        return frame, valid_actions
    
    def step(self, action, frame, *args, **kwargs):
        """
        Executes an action (as defined in the Environment class) with the env.step function.
        """
        torch_frame = torch.tensor(frame, device=self.device).unsqueeze(0)
        torch_action = torch.tensor([action], device=self.device)
        latent = self.prior.sample(torch_frame, torch_action) 
        latent = int(latent.item()) # from torch tensor of size 1 to scalar integer
        #print("Sampled latent: ", latent)
        self.env.set_state(frame)
        frame, reward, done = self.env.step([action,latent], *args, **kwargs)
        valid_actions = np.arange(self.action_space) # TODO: make this be returned from the env.step method
        return frame, valid_actions, reward, done
    
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
            
            