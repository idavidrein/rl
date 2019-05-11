import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
import numpy as np

def discrete_network(dims = None):

	x = Input(shape = dims[0], name = "input")
	hidden = Dense(32, activation = 'relu', name = 'hidden')(x)
	output = Dense(dims[1], activation = 'softmax', name = 'output')(hidden)
	policy = Model(inputs = x, outputs = output)

	return policy



# class continuous_network():