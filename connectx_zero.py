import numpy as np
from collections import deque
import connectx
from mcts import MCTS
from nnet_torch import Policy

class ConnectXAgent():
    def __init__(
            self,
            env,
            n_sims_train=10,
            c_puct=1,
            device='cpu'
    ):
        self.env = env
        self.n_sims_train = n_sims_train
        self.c_puct = c_puct
        self.config = env.configuration
        self.policy = Policy(device)

    def train(self, n_iters=10, n_eps=100, max_memory=1000):
        examples = deque(maxlen=max_memory)
        for i in range(n_iters):
            for e in range(n_eps):
                examples.extend(self.execute_episode())
            self.policy.train(examples)
        
    def execute_episode(self):
        examples = []
        s = connectx.empty_grid(self.config)
        tree = MCTS(s, self.env, self.policy, self.c_puct)
        mark = 1

        while True:
            for _ in range(self.n_sims_train):
                tree.search(s)
            action_probs = tree.pi(s)
            examples.append([s, action_probs])
            a = np.random.choice(len(action_probs), p=action_probs)
            s = connectx.drop_piece(s, a, 1, self.config)
            # backup scores on game end
            if connectx.is_terminal_grid(s, self.config):
                reward = connectx.score_game(s, self.config)
                for ex in examples:
                    ex.append(reward * mark)
                    mark *= -1
                return examples
            else: # swap board perspective
                s = connectx.reverse_grid(s)
                mark *= -1

    def choose_move(self, board, mark, n_sims=10, deterministic=True):
        board = [-1 if token == 2 else token for token in board]
        board = np.reshape(board, (self.config.rows, self.config.columns))
        if mark == 2:
            board = -board
        s = tuple(map(tuple, board))
        tree = MCTS(s, self.env, self.policy, self.c_puct)
        for _ in range(n_sims):
            tree.search(s)
        if deterministic:
            return tree.best_action(s)
        else:
            return tree.stochastic_action(s)
        
    def load_policy(self, state_dict):
        self.policy.nnet.load(state_dict)