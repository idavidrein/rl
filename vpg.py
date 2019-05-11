import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from utils import discrete_network, get_dims

def vpg(environment='CartPole-v0', hidden_units=32, gamma=0.8, 
        seed_num=10, learning_rate=.01, num_episodes=400,
        batch_size=10, max_steps=1000, num_layers = 1):

    # ensure reproducibility (and make debugging possible)
    np.random.seed(seed_num)
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    tf.random.set_seed(seed_num)
    
    env = gym.make(environment)

    obs_dim, action_dim = get_dims(env.action_space, env.observation_space)

    # create policy network
    # to-do: only works for Discrete, fix for continuous!
    policy = discrete_network(dims = (obs_dim, action_dim))
    print()
    policy.summary()
    
    optimizer = tf.keras.optimizers.Adam(lr = learning_rate)

    # create buffer so we can sum gradients
    grad_buffer = policy.trainable_variables
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0

    rewards = []

    print("\nRunning policy in environment and training...\n")

    # main execution loop
    for ep_number in range(num_episodes):
        
        ep_reward = 0
        obs = np.array(env.reset()).reshape(1, -1)
        
        ep_buffer = []

        for j in range(max_steps):
            # env.render()
            with tf.GradientTape() as tape:

                # take action
                # to-do: only works for Discrete, fix for continuous!
                action_probs = policy(obs)
                log_probs = tf.math.log(action_probs)
                action = int(tf.random.categorical(log_probs, 1))
                log = log_probs[0, action]

            # take gradient w.r.t. params of log of action taken
            grads = tape.gradient(log, policy.trainable_variables)
            obs, reward, done, info = env.step(action)
            obs = np.array(obs).reshape(1, -1)

            # record info in buffer
            ep_buffer.append([reward, grads])
            ep_reward += reward

            if done:
                break

        rewards.append(ep_reward)

        ep_buffer = np.array(ep_buffer)

        # compute rewards-to-go
        # to-do: advantage function???
        rewards_to_go = np.zeros(ep_buffer.shape[0])
        for t in range(ep_buffer.shape[0]):
            length = ep_buffer.shape[0] - t
            weights = np.array([gamma ** i for i in range(length)])
            rewards_to_go[t] = np.sum(np.multiply(ep_buffer[t:,0], weights))

        # add episode information to current estimation of policy gradient
        for t, ep_info in enumerate(ep_buffer):
            for ix, grad in enumerate(ep_info[1]):
                grad_buffer[ix] -= (1 / batch_size) * grad * rewards_to_go[t]

        # every batch_size number of episodes, 
        # run gradient descent on sample
        if ep_number % batch_size == 0:
            print("Episode {0} reward: {1}".format(ep_number, np.mean(rewards[-batch_size:])))

            optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))

            # re-initialize buffer to zero (on-policy)
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=32)
    # use for # of hidden layers
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--eps', type=int, default=50)
    # use for logging
    # to-do: implement good logging procedure; can take inspiration from openai (medium)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    vpg(environment=args.env, hidden_units=args.hid, gamma=args.gamma, 
        seed_num=args.seed, learning_rate=args.lr, num_episodes=args.eps,
        batch_size=args.batch, num_layers=args.l)