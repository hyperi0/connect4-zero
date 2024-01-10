import numpy as np
import keras
from keras import layers

class PolicyNet():
    def __init__(
            self,
            input_shape,
            num_actions,
            epochs=10,
            batch_size=64
    ):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.epochs = epochs
        self.batch_size = batch_size
        self.nnet = None
        self.init_net()

    def init_net(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Flatten()(x)
        action_probs = layers.Dense(self.num_actions, activation="softmax")(x)
        value = layers.Dense(1)(x)
        outputs = {"pi": action_probs, "v": value}
        nnet = keras.Model(inputs=inputs, outputs=outputs)
        nnet.compile(
            loss={
                "pi": keras.losses.CategoricalCrossentropy(),
                "v": keras.losses.MeanSquaredError()
            }
        )
        self.nnet = nnet

    def learn(self, examples):
        s, pi, v = np.transpose(examples)
        self.nnet.fit(
            x=s,
            y={"pi": pi, "v": v},
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def predict(self, s):
        return self.nnet(np.asarray(s))