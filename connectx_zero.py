import numpy as np
from collections import deque
import connectx
from mcts import MCTS
from nnet import PolicyNet

class ConnectXAgent():
    def __init__(
            self,
            env,
            n_sims_train,
            c_puct,
    ):
        self.env = env
        self.n_sims_train = n_sims_train
        self.c_puct = c_puct
        self.config = env.configuration
        self.nnet = None

    def train(self, n_iters=10, n_eps=100, max_memory=1000):
        self.nnet = PolicyNet(
            input_shape = (self.config.rows, self.config.columns, 1),
            num_actions = self.config.columns
        )
        examples = deque(maxlen=max_memory)
        for i in range(n_iters):
            for e in range(n_eps):
                examples.append(self.executeEpisode())
            self.nnet.learn(examples)
        
    def execute_episode(self):
        examples = []
        s = connectx.empty_grid(self.config)
        tree = MCTS(s, self.env, self.nnet, self.c_puct)
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