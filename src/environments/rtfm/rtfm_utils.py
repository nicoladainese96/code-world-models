from rtfm import featurizer as X
from rtfm import tasks # needed to make rtfm visible as Gym env
import rtfm.render_4by4 as rtfm_render
import gym
import numpy as np
import torch
import revtok

action_dict = {
        0:"Stay",
        1:"Up",
        2:"Down",
        3:"Left",
        4:"Right"
    }

# simulate flags without argparse
class Flags():
    def __init__(self, 
                 env="rtfm:groups_simple_stationary-v0", # simplest env
                 #env="rtfm:groups-v0",
                 height=6,
                 width=6,
                 partial_observability=False,
                 max_placement=2,
                 shuffle_wiki=False,
                 time_penalty=0
                 ):
        self.env = env
        self.height = height
        self.width = width
        self.partial_observability = partial_observability
        self.max_placement = max_placement
        self.shuffle_wiki = shuffle_wiki 
        self.time_penalty = time_penalty

def create_env(flags, featurizer=None):
    f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
    env = gym.make(flags.env, 
                   room_shape=(flags.height, flags.width), 
                   partially_observable=flags.partial_observability, 
                   max_placement=flags.max_placement, 
                   featurizer=f, 
                   shuffle_wiki=flags.shuffle_wiki, 
                   time_penalty=flags.time_penalty)
    return env

def init_gym_env_and_wrapper(env="rtfm:groups_simple-v0", 
                 **kwargs):
    flags = Flags(env=env, **kwargs)
    gym_env = create_env(flags)
    env = TrueSimulatorWithGroundTruthClasses(gym_env)
    return gym_env, env

def process_frame(s, device):
    s_t = {}
    for k in s.keys():
        if k=='name':
            s_t[k] = s[k][None,:,1:-1,1:-1,...].to(device)
        else:
            s_t[k] = s[k][None,...].to(device)
    return s_t

def render_log_probs(frame_lp, inv_lp, initial_frame, env):
    pred_frame = frame_lp.argmax(dim=-1)
    pred_inv = inv_lp.argmax(dim=-1)
    s = initial_frame
    frame_tp1 = {'name':pred_frame, 'inv':pred_inv, 'wiki':s['wiki'], 'task':s['task']}
    env.render_frame(frame_tp1)
    
class TextRTFM():
    def __init__(self, env, featurizer=None):
        self.env = env
        self.action_space = len(env.action_space)
        self.featurizer = featurizer
        
    def _frame_to_words(self, x):
        x_words = np.zeros((6,6,2), dtype=object) 
        for i in range(6):
            for j in range(6):
                for k in range(2):
                    w1 = self.env.vocab.index2word(x[i,j,k,0])
                    w2 = self.env.vocab.index2word(x[i,j,k,1])
                    if w2 != 'pad':
                        full_w = w1+' '+w2
                    else:
                        full_w = w1
                    x_words[i,j,k] = full_w
        return np.array(x_words)
    
    def _inv_to_words(self, x):
        w1 = self.env.vocab.index2word(x[0]) 
        if w1 == 'pad':
            w1 = 'empty'
        w2 = self.env.vocab.index2word(x[1]) 
        if w2 != 'pad':
            full_w = w1+' '+w2
        else:
            full_w = w1
        return full_w
        
    def _process_frame(self, frame):
       
        for k in frame.keys():
            frame[k] = frame[k].numpy()
            
        # do this before batch dim is added
        valid_moves = frame['valid'].astype(bool) # boolean mask of shape (action_space)
        actions = np.arange(self.action_space)
        valid_actions = actions[valid_moves]
        
        target_classes = self.get_ground_truth_roles()
        frame['trg_cls'] = target_classes
        
        keys = ['name_len', 'inv_len', 'wiki', 'wiki_len', 'task', 'task_len', 'valid', 'trg_cls']
        extra_info = {}
        for k in keys:
            extra_info[k] = frame[k]
            
        frame = (self._frame_to_words(frame['name'][:,:,:,:2]), self._inv_to_words(frame['inv'][:2]))
        return frame, valid_actions, extra_info
    
    def reset(self):
        frame = self.env.reset()
        frame, valid_actions, extra_info = self._process_frame(frame)
        return frame, valid_actions, extra_info
    
    def step(self, action, *args, **kwargs):
        frame, reward, done, _ = self.env.step(int(action), *args, **kwargs)
        frame, valid_actions, extra_info = self._process_frame(frame)
        return frame, valid_actions, reward, done, extra_info
    
    def render(self):
        self.featurizer.featurize(self.env)
    
    def render_frame(self, frame, extra_info):
        f = {}
        for k in frame.keys():
            f[k] = frame[k].squeeze()
        rtfm_render.render_frame(f, self.env)
        
    def save_state_dict(self):
        return self.env.save_state_dict()
        
    def load_state_dict(self, d):
        self.env.load_state_dict(d)
        
    def get_ground_truth_roles(self):
        # get tokens from gym_env
        target_convention = {'!':0, '?':1, 'y':2, 'n':3}
        target = torch.zeros((4,2))
        entities = list(self.env.world.monsters - self.env.world.agents) +\
            list(self.env.world.items) + list(self.env.agent.inventory.equipped.values())

        for e in entities:
            position = target_convention[e.char]
            token = self.lookup_sentence(e.describe(), self.env.vocab, max_len=self.env.max_name)[0][:-1]
            #print(e.name, token)
            target[position] = torch.tensor(token)
        return target.long()
    
    def lookup_sentence(self, sent, vocab, max_len=10, eos='pad', pad='pad'):
        """
        Requires import revtok
        """
        if isinstance(sent, list):
            words = sent[:max_len-1] + [eos]
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        else:
            #print('sent ', sent)
            sent = sent.lower()
            key = sent, max_len
            words = revtok.tokenize(sent)[:max_len-1] + [eos]
            #print('words', words)
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        
class TrueSimulatorWithGroundTruthClasses():
    """
    Returns the full state from the environment step.
    """
    def __init__(self, env, featurizer=None):
        self.env = env
        self.action_space = len(env.action_space)
        self.featurizer = featurizer
        
    def _process_frame(self, frame):
        """
        Extracts from frame a valid_action numpy array containing integers from 0 to self.action_space-1
        Adds batch dimension to all values stored inside frame dictionary
        """
        # do this before batch dim is added
        valid_moves = frame['valid'].numpy().astype(bool) # boolean mask of shape (action_space)
        actions = np.arange(self.action_space)
        valid_actions = actions[valid_moves]
        
        target_classes = self.get_ground_truth_roles()
        
        for k in frame.keys():
            frame[k] = frame[k].unsqueeze(0)
        
        frame['trg_cls'] = target_classes.unsqueeze(0)
        
        return frame, valid_actions
    
    def reset(self):
        frame = self.env.reset()
        frame, valid_actions = self._process_frame(frame)
        return frame, valid_actions
    
    def step(self, action, *args, **kwargs):
        frame, reward, done, _ = self.env.step(int(action), *args, **kwargs)
        frame, valid_actions = self._process_frame(frame)
        return frame, valid_actions, reward, done
    
    def render(self):
        self.featurizer.featurize(self.env)
    
    def render_frame(self, frame):
        f = {}
        for k in frame.keys():
            f[k] = frame[k].squeeze()
        rtfm_render.render_frame(f, self.env)
        
    def save_state_dict(self):
        return self.env.save_state_dict()
        
    def load_state_dict(self, d):
        self.env.load_state_dict(d)
        
    def get_ground_truth_roles(self):
        # get tokens from gym_env
        target_convention = {'!':0, '?':1, 'y':2, 'n':3}
        target = torch.zeros((4,2))
        entities = list(self.env.world.monsters - self.env.world.agents) +\
            list(self.env.world.items) + list(self.env.agent.inventory.equipped.values())

        for e in entities:
            position = target_convention[e.char]
            token = self.lookup_sentence(e.describe(), self.env.vocab, max_len=self.env.max_name)[0][:-1]
            #print(e.name, token)
            target[position] = torch.tensor(token)
        return target.long()
    
    def lookup_sentence(self, sent, vocab, max_len=10, eos='pad', pad='pad'):
        """
        Requires import revtok
        """
        if isinstance(sent, list):
            words = sent[:max_len-1] + [eos]
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        else:
            #print('sent ', sent)
            sent = sent.lower()
            key = sent, max_len
            words = revtok.tokenize(sent)[:max_len-1] + [eos]
            #print('words', words)
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        
def print_board(board, logger_function=None):
    board_str = ""
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            print_str = str(board[i, j, 0])
            if board[i, j, 1] != 'empty':
                print_str += f"({board[i, j, 1]})"
            board_str += print_str.ljust(25)
        board_str += "\n"

    if logger_function is not None:
        logger_function(board_str)
    else:
        print(board_str)

action_dict = {0: "Stay", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}

def print_trajectory(trajectory, logger_function=None):
    if logger_function is not None:
        print = logger_function
    states, actions, next_states, rewards, dones, extra_infos = trajectory
    for i in range(len(states)):
        print(f"Step {i}")
        print_board(states[i][0], logger_function)
        print("Inventory: ", states[i][1])
        print("\n")
        print("Action: ", action_dict[actions[i]])
        print("Reward: ", rewards[i])
        print("Done: ", dones[i])
        print("\n")

def print_transition(transition, logger_function=None):
    if logger_function is not None:
        print = logger_function
    state, action, next_state, reward, done, extra_info = transition
    print_board(state[0][0], logger_function)
    print("Inventory: ", state[0][1])
    print("\n")
    print("Action: ", action_dict[action[0]])
    print("\n")
    print_board(next_state[0][0], logger_function)
    print("Inventory: ", next_state[0][1])
    print("Reward: ", reward[0])
    print("Done: ", done[0])