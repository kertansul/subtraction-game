import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_agents.networks import network


class QNetwork(network.Network):

    def __init__(
        self,
        input_tensor_spec,
        action_spec,
        depth=5,
        num_hidden_units=256,
        name=None
    ):
        input_tensor_spec = input_tensor_spec
        action_spec = action_spec
        num_actions = action_spec.maximum - action_spec.minimum + 1
        super(QNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self.backbone = []
        for _ in range(depth-1):
            self.backbone.append(keras.layers.Dense(num_hidden_units, activation='tanh'))
        self.backbone.append(keras.layers.Dense(num_actions, activation='linear'))

    def call(self, inputs, step_type=None, network_state=()):
        del step_type
        inputs = tf.cast(inputs, tf.float32)
        # flat
        x = keras.layers.Flatten()(inputs)  # (B, 9)
        # extract features
        _index = tf.constant([
            [0, 3, 6], [0, 4, 8], [0, 5, 7],
            [1, 4, 7], [1, 3, 8], [1, 5, 6],
            [2, 5, 8], [2, 4, 6], [2, 3, 7],
            [0, 1, 2], [3, 4, 5], [6, 7, 8]])
        feats = tf.gather(x, _index, axis=1)  # (B, 12, 3)
        min_feats = tf.reduce_min(feats, axis=2)  # (B, 12)
        max_feats = tf.reduce_max(feats, axis=2)  # (B, 12)
        onetwothree = tf.broadcast_to(tf.range(1, 4, dtype=tf.float32), [tf.shape(x)[0], 3])  # (B, 3)
        x = tf.concat([x, min_feats, max_feats, onetwothree], axis=1)  # (B, 9+12+12+3)
        # build network
        for layer in self.backbone:
            x = layer(x)
        return x, network_state

    def save_weights(self, dir, step=0):
        if not tf.io.gfile.exists(dir):
            tf.io.gfile.makedirs(dir)
        for i, layer in enumerate(self.backbone):
            dense, bias = layer.get_weights()
            np.save(f'{dir}/step{step:07d}_layer{i}_dense.npy', dense)
            np.save(f'{dir}/step{step:07d}_layer{i}_bias.npy', bias)

    def load_weights(self, dir, step=0):
        for i, layer in enumerate(self.backbone):
            dense = np.load(f'{dir}/step{step:07d}_layer{i}_dense.npy')
            bias = np.load(f'{dir}/step{step:07d}_layer{i}_bias.npy')
            layer.set_weights([dense, bias])