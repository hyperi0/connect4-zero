from mcts import MCTS
import connectx
import random
import numpy as np
import keras
from keras import layers
from collections import deque

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
        self.init_net(
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
        mark = 1
        tree = MCTS(s, self.env, self.nnet, self.c_puct)
        
        while True:
            for _ in range(self.n_sims_train):
                tree.search(s, mark)
            action_probs = tree.pi(s)
            examples.append([s, action_probs])
            a = random.choices(range(self.config.columns), weights=action_probs)
            s = connectx.drop_piece(s, a, mark, self.config)
            if connectx.is_terminal_grid(s, self.config):
                reward = connectx.score_game(s, self.config)
                for ex in examples:
                    ex.append(reward)
                return examples

    def init_net(self, input_shape, num_actions):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Flatten()(x)
        action_probs = layers.Dense(num_actions, activation="softmax")(x)
        value = layers.Dense(1)(x)
        outputs = layers.Concatenate()([action_probs, value])
        self.nnet = keras.Model(inputs=inputs, outputs=outputs)