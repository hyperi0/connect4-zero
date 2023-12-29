import mcts
import numpy as np
import keras
from keras import layers

class connectx_agent():
    def __init__():
        pass

    def execute_episode():
        pass

    def train(self, n_iters, n_eps, max_memory):
        pass

class connectx_cnn():
    def __init__(
            self,
            input_shape,
            num_actions,
    ):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.nnet = None
    
    def init_net(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Flatten()(x)
        action_probs = layers.Dense(self.num_actions, activation="softmax")(x)
        value = layers.Dense(1)(x)
        outputs = layers.Concatenate()([action_probs, value])

        self.nnet = keras.Model(inputs=inputs, outputs=outputs)