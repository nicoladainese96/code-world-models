import numpy as np
import copy
import torch
import torch.nn.functional as F
import hashlib

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.WARN)

from .mcts import StochasticStateNode, StochasticStateActionNode, StochasticMCTS

def hash_frame(frame):
    h = hashlib.sha256()
    h.update(str(frame).encode('utf-8'))
    return h.hexdigest()

def print_board(board, logger):
    board_str = ""
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            print_str = str(board[i, j, 0])
            if board[i, j, 1] != 'empty':
                print_str += f"({board[i, j, 1]})"
            board_str += print_str.ljust(25, end="")
        board_str += "\n"

    logger.info(board_str)

class StochasticMCTS_CWM(StochasticMCTS):
    def __init__(self,
                 root_frame,
                 simulator,
                 valid_actions,
                 ucb_c,
                 discount,
                 max_actions,
                 root=None,
                 render=False,
                 logger=DEFAULT_LOGGER):
        """
        Monte Carlo Tree Search assuming stochastic dynamics.
        
        root_frame: dict of torch tensors
            Processed frame of the current state (a.k.a. root node).
        simulator: object
            Instance of StochasticCWMSimulator from ./environments/simulator
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
        super().__init__(root_frame, simulator, valid_actions, ucb_c, discount, max_actions, root, render, logger)
    

    def simulator_step(self, simulator, action, frame):
        return simulator.step(action, frame)