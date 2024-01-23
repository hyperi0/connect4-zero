import numpy as np
from collections import deque
from connectx import *
from mcts import MCTS
from nnet_torch import Policy
from tqdm import trange
import time

class ConnectXAgent():
    def __init__(
            self,
            config,
            n_sims_train=10,
            c_puct=1,
            device='cuda',
    ):
        self.config = config
        self.n_sims_train = n_sims_train
        self.c_puct = c_puct
        self.policy = Policy(device)
        self.tree = MCTS(self.config, self.policy, self.c_puct)
        self.examples = []

    def train(self, n_iters=10, n_eps=100, max_memory=1000):
        examples = deque(maxlen=max_memory)
        for i in range(n_iters):
            with trange(n_eps, unit='eps') as pbar:
                pbar.set_description(f'Iteration {i} of {n_iters}')
                for e in pbar:
                    examples.extend(self.execute_episode())
            self.policy.train(examples)
        
    def execute_episode(self):
        examples = []
        s = empty_grid(self.config)
        tree = MCTS(self.config, self.policy, self.c_puct)
        mark = 1

        while True:
            for _ in range(self.n_sims_train):
                tree.search(s)
            action_probs = tree.pi(s)
            examples.append([s, action_probs])
            a = np.random.choice(len(action_probs), p=action_probs)
            s = drop_piece(s, a, 1, self.config)
            # backup scores on game end
            if is_terminal_grid(s, self.config):
                reward = score_game(s, self.config)
                for ex in examples:
                    ex.append(reward * mark)
                    mark *= -1
                self.examples = examples
                return examples
            else: # swap board perspective
                s = reverse_grid(s)
                mark *= -1

    def choose_move(self, board, mark, max_time=1.5, deterministic=True):
        board = [-1 if token == 2 else token for token in board]
        board = np.reshape(board, (self.config.rows, self.config.columns))
        if mark == 2:
            board = -board
        s = tuple(map(tuple, board))
        start_time = time.time()
        while time.time() - start_time < max_time:
            self.tree.search(s)
        if deterministic:
            return self.tree.best_action(s)
        else:
            return self.tree.stochastic_action(s)
        
    def move_probabilities_values(self, board, mark, n_sims=10):
        board = [-1 if token == 2 else token for token in board]
        board = np.reshape(board, (self.config.rows, self.config.columns))
        if mark == 2:
            board = -board
        s = tuple(map(tuple, board))
        for _ in range(n_sims):
            self.tree.search(s)
        info = {
            "P": self.tree.P[s],
            "Q": self.tree.Q[s],
            "N": self.tree.N[s],
            "pi": self.tree.pi(s)
        }
        return info

    def load_policy(self, state_dict):
        self.policy.nnet.load_state_dict(state_dict)