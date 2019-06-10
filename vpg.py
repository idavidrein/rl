import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from utils import *
from gym.spaces import Discrete, Box
import atexit


def vpg(environment='CartPole-v0', hidden_units=32, gamma=0.9,
        seed_num=10, learning_rate=.01, num_episodes=400,
        batch_size=10, max_steps=1000, num_layers=1,
        fpath="logs/log", arg_dict=dict(), lam=.9, save=''):
    logger = Logger(file_name=fpath, info=arg_dict)
    save_path = "models/" + save
    policy_path = save_path + "_policy"

    # ensure reproducibility (and make debugging possible)
    np.random.seed(seed_num)
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    tf.random.set_seed(seed_num)

    env = gym.make(environment)

    obs_dim, action_dim = get_dims(env.action_space, env.observation_space)

    # create policy network (actor)
    # TODO: check policy to ensure it's created properly
    policy = create_policy(env.action_space, obs_dim, action_dim)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    # create buffer so we can aggregate gradients
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

        # run one episode
        for j in range(max_steps):
            with tf.GradientTape() as tape:

                # run policy on observation
                if isinstance(env.action_space, Discrete):
                    action_probs = policy(obs)
                    log_probs = tf.math.log(action_probs)
                    # TODO: check actions are chosen properly
                    action = int(tf.random.categorical(log_probs, 1))
                    log = log_probs[0, action]
                else:
                    action, log = policy(obs)

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

        # compute discounted rewards-to-go
        # TODO: check rewards-to-go to ensure they're calculated correctly
        num_steps = ep_buffer.shape[0]
        rewards_to_go = np.zeros(num_steps)
        for t in range(num_steps):
            length = num_steps - t
            weights = np.ones(length) * gamma
            # create discounting array
            weights = np.power(weights, range(length))
            # calculate rewards-to-to
            rewards_to_go[t] = np.sum(np.multiply(weights, ep_buffer[t:, 0]))

        # add episode information to current estimation of policy gradient
        for t, ep_info in enumerate(ep_buffer):
            for ix, grad in enumerate(ep_info[1]):
                # subtract because optimizer minimizes loss function,
                # and we want to maximize expected reward
                grad_buffer[ix] -= (1 / batch_size) * grad * rewards_to_go[t]

        # every batch_size number of episodes, 
        # run gradient descent on samples
        if ep_number % batch_size == 0 and ep_number != 0:

            stats = summary_stats(np.array(rewards[-batch_size:]))
            logger.log_epoch(reward=stats)

            # policy optimization
            optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))
            # re-initialize buffer to zero (on-policy)
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0

            print("Epoch {0} reward: {1}".format(
                ep_number / batch_size,
                np.mean(rewards[-batch_size:])))

        # if ctrl+c is used to stop training, the models will be saved
        if save:
            atexit.register(save_models, [policy], [policy_path])
    if save:
        # TODO: figure out issue with the saved model not being compiled
        # even though it should be >:( and implement functionality to 
        # save the model every so often and be able to resume training
        # given just a saved model
        save_models([policy], [policy_path])
        print("Saved!")


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
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--eps', type=int, default=2000)
    parser.add_argument('--lam', type=float, default=.97)
    parser.add_argument('--save', type=str, default='')
    # use for logging
    # to-do: implement good logging procedure; can take inspiration from openai (medium)
    parser.add_argument('--exp_name', type=str, default='log1')
    args = parser.parse_args()

    vpg(environment=args.env, hidden_units=args.hid, gamma=args.gamma,
        seed_num=args.seed, learning_rate=args.lr, num_episodes=args.eps,
        batch_size=args.batch, num_layers=args.l, fpath=args.exp_name,
        arg_dict=vars(args), lam=args.lam, save=args.save)
