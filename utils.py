import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
import numpy as np
from gym.spaces import Box, Discrete
import gym
import json


def discrete_network(dims=None, output_activation='softmax'):
    x = Input(shape=dims[0], name="input")
    hidden = Dense(4, activation='relu', name='hidden_1')(x)
    output = Dense(dims[1], activation=output_activation, name='output')(hidden)
    model = Model(inputs=x, outputs=output)
    return model


def continuous_network(dims=None):
    x = Input(shape=dims[0], name="input")
    hidden = Dense(4, activation='relu', name='hidden')(x)
    means = Dense(dims[1], name='means')(hidden)
    action, likelihood = sampling()(means)
    model = Model(inputs=x, outputs=[action, likelihood])
    return model


class sampling(tf.keras.layers.Layer):
    def __init__(self):
        super(sampling, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[input_shape[-1]],
                                        initializer=tf.initializers.ones(), dtype=tf.float32)

    def call(self, x):
        exp = tf.exp(self.kernel)
        action = x + tf.random.normal([int(x.shape[-1])]) * exp
        likelihood = log_likelihood(action, x, self.kernel)
        return action, likelihood


def log_likelihood(x, mu, log_std):
    likelihood = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return likelihood


def kl(p, q):
    return -np.sum(p * np.log(q / p))


def save_models(models, filepaths):
    assert (len(models) == len(filepaths))
    for ix in range(len(models)):
        models[ix].save(filepaths[ix])


class Logger():
    def __init__(self, file_name, info):
        self.fpath = 'logs/' + file_name
        self.output_file = open(self.fpath, 'w')
        self.info = info

        with open(self.fpath + '_info.json', 'w') as info_file:
            json.dump(self.info, info_file, indent=4)

    def log_epoch(self, **kwargs):
        return


def summary_stats(info):
    summaries = dict()
    summaries['sum'] = info.sum()
    summaries['mean'] = np.mean(info)
    summaries['min'] = info.min()
    summaries['max'] = info.max()
    summaries['variance'] = np.var(info)
    return summaries


def create_policy(action_space, obs_dim, action_dim):
    if isinstance(action_space, Discrete):
        policy = discrete_network(dims=(obs_dim, action_dim))
    else:
        policy = continuous_network(dims=(obs_dim, action_dim))
    return policy


def get_dims(act_space, obs_space):
    if isinstance(act_space, Discrete):
        action_dim = act_space.n

    else:
        action_dim = act_space.shape[0]

    if isinstance(obs_space, Discrete):
        obs_dim = (1,)

    elif isinstance(obs_space, gym.spaces.Tuple):
        obs_dim = len(obs_space)

    else:
        obs_dim = obs_space.shape

    return obs_dim, action_dim


if __name__ == '__main__':
    layer = sampling()
    print(layer(np.array([[10]], dtype=np.float32)))
