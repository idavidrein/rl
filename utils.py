import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
import numpy as np
from gym.spaces import Box, Discrete
import gym

def discrete_network(dims = None):

    x = Input(shape = dims[0], name = "input")
    hidden = Dense(32, activation = 'relu', name = 'hidden')(x)
    output = Dense(dims[1], activation = 'softmax', name = 'output')(hidden)
    policy = Model(inputs = x, outputs = output)

    return policy

def continuous_network(dims = None):
    

def get_dims(act_space, obs_space):

    if isinstance(act_space, Discrete):
        action_dim = act_space.n

    else: action_dim = act_space.shape

    if isinstance(obs_space, Discrete):
        obs_dim = (1,)

    elif isinstance(obs_space, gym.spaces.Tuple):
        obs_dim = len(obs_space)

    else: obs_dim = obs_space.shape

    return obs_dim, action_dim