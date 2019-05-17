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
    x = Input(shape = dims[0], name = "input")
    hidden = Dense(32, activation = 'relu', name = 'hidden')(x)
    means = Dense(dims[1], name = 'means')(hidden)
    action, likelihood = sampling(dims[1])(means)
    policy = Model(inputs = x, outputs = [action, likelihood])
    
    return policy

class sampling(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(sampling, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", 
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer=tf.initializers.ones(), dtype = tf.float32)

    def call(self, input):
        exp = tf.exp(self.kernel)
        action = input + tf.random.normal([input.shape[1]]) * exp
        likelihood = log_likelihood(action, input, self.kernel)
        return action, likelihood



def log_likelihood(x, mu, log_std):
	likelihood = -0.5*(((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
	return likelihood

def get_dims(act_space, obs_space):

    if isinstance(act_space, Discrete):
        action_dim = act_space.n

    else: action_dim = act_space.shape[0]

    if isinstance(obs_space, Discrete):
        obs_dim = (1,)

    elif isinstance(obs_space, gym.spaces.Tuple):
        obs_dim = len(obs_space)

    else: obs_dim = obs_space.shape

    return obs_dim, action_dim


if __name__ == '__main__':
    layer = sampling(num_outputs = 1)
    print(layer(np.array([[10]], dtype = np.float32)))