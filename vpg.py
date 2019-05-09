import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

def vpg(environment='CartPole-v0', hidden_units=32, gamma=0.8, 
    	seed_num=10, learning_rate=.01, num_episodes=400,
    	batch_size=10, max_steps=1000):
	
	env = gym.make(environment)

	# to-do: only works for Discrete, fix for continuous!
	action_dim = env.action_space.n
	obs_dim = len(env.observation_space.high)

	policy = tf.keras.Sequential([
    	tf.keras.layers.Dense(hidden_units, input_shape = (obs_dim,), activation = 'relu'),
    	tf.keras.layers.Dense(action_dim, activation = 'softmax')
    ])
	


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=32)
    # use for # of hidden layers
    # to-do: implement loop of hidden layers (easy)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--eps', type=int, default=50)
    # use for logging
    # to-do: implement good logging procedure; can take inspiration from openai (medium)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    vpg(environment=args.env, hidden_units=args.hid, gamma=args.gamma, 
    	seed_num=args.seed, learning_rate=args.lr, num_episodes=args.eps,
    	batch_size=args.batch)