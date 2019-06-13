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


def ppo(environment='CartPole-v0', hidden_units=32, gamma=0.9,
        seed_num=10, learning_rate=.01, num_episodes=400,
        batch_size=10, max_steps=1000, num_layers=1, epsilon=0.2,
        fpath="logs/log", arg_dict=dict(), lam=.9, save=''):
    logger = Logger(file_name=fpath, info=arg_dict)
    save_path = "models/" + save
    policy_path = save_path + "_policy"
    critic_path = save_path + "_critic"

    # ensure reproducibility (and make debugging possible)
    np.random.seed(seed_num)
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    tf.random.set_seed(seed_num)

    env = gym.make(environment)

    obs_dim, action_dim = get_dims(env.action_space, env.observation_space)

    # create policy network (actor)
    policy = create_policy(env.action_space, obs_dim, action_dim)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = 0

    # create value network (critic)
    critic = discrete_network(dims=(obs_dim, 1), output_activation=None)
    critic.compile(optimizer='adam', loss='mean_squared_error')
    critic_buffer = []
    val_loss = []

    # create buffer so we can aggregate gradients
    grad_buffer = policy.trainable_variables
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0

    rewards = []

    print("\nRunning policy in environment and training...\n")

    # main execution loop
    for ep_number in range(num_episodes):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(policy.trainable_variables)
            critic_buffer.append([])

            ep_reward = 0
            obs = np.array(env.reset()).reshape(1, -1)

            ep_buffer = []

            # run one episode
            for j in range(max_steps):
                # run policy on observation
                # print(obs.shape)
                if isinstance(env.action_space, Discrete):
                    action_probs = policy(obs)
                    log_probs = tf.math.log(action_probs)
                    action = int(tf.random.categorical(log_probs, 1))
                    log = log_probs[0, action]
                else:
                    action, log = policy(obs)

                prev_obs = obs

                # take gradient w.r.t. params of log of action taken
                obs, reward, done, info = env.step(action)
                obs = np.array(obs).reshape(1, -1)

                # record info in buffer
                ep_buffer.append([reward, log])
                ep_reward += reward

                # fill critic info buffer
                critic_buffer[-1].append(prev_obs)

                if done:
                    break

            rewards.append(ep_reward)

            ep_rewards = np.array([step[0] for step in ep_buffer])

            critic_buffer[-1] = np.reshape(critic_buffer[-1], (-1, obs_dim[0]))

            # compute discounted rewards-to-go and advantage function
            # uses Generalized Advantage Estimation (GAE)
            # TODO: modularize GAE into function
            num_steps = ep_rewards.shape[0]
            rewards_to_go = np.zeros(num_steps)
            advantages = np.zeros(num_steps)
            val_estimates = critic(critic_buffer[-1])
            for t in range(num_steps):
                length = num_steps - t
                weights = np.ones(length) * gamma
                # create discounting array
                weights = np.power(weights, range(length))
                # calculate rewards-to-to
                rewards_to_go[t] = np.sum(np.multiply(weights, ep_rewards[t:]))
                weights = weights * lam
                for l in range(length):
                    # if-else is necessary to avoid buffer overflow
                    if l < length - 1:
                        # calculate TD-Residual
                        delta = ep_rewards[t + l] + lam * val_estimates[t + l + 1] - val_estimates[t + l]
                        advantages[t] += weights[l] * delta
                    else:
                        # end of trajectory, so V(s_(t+1)) = 0
                        delta = ep_rewards[t + l] - val_estimates[t + l]
                        advantages[t] += weights[l] * delta

            # attach rewards to critic buffer
            critic_buffer[-1] = np.concatenate(
                (critic_buffer[-1], np.expand_dims(rewards_to_go, axis=1)),
                axis=1)

            # add episode information to current estimation of loss
            for t, ep_info in enumerate(ep_buffer):
                # subtract because optimizer minimizes loss function,
                # and we want to maximize the PPO-Clip objective
                action_log_prob = ep_info[1]
                old_log_prob = tf.stop_gradient(action_log_prob)
                importance_ratio = tf.exp(action_log_prob - old_log_prob)
                adv = advantages[t]
                loss -= tf.math.minimum(importance_ratio * adv, g(epsilon, advantages[t]) * adv)
            loss *= 1 / (batch_size * num_steps)

        grads = tape.gradient(loss, policy.trainable_variables)
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] += grads[ix]

        # every batch_size number of episodes,
        # run gradient descent on samples
        if ep_number % batch_size == 0 and ep_number != 0:
            stats = summary_stats(np.array(rewards[-batch_size:]))
            logger.log_epoch(reward=stats)

            # policy optimization
            optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))
            # re-initialize loss and grad_buffer to zero (on-policy)
            loss = 0 * loss
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0

            # value function optimization
            critic_buffer_arr = np.concatenate(critic_buffer)
            history = critic.fit(
                x=critic_buffer_arr[:, :-1],
                y=critic_buffer_arr[:, -1],
                batch_size=critic_buffer_arr.shape[0],
                verbose=0)

            print("Epoch {0} reward: {1}, value loss: {2}".format(
                ep_number / batch_size,
                np.mean(rewards[-batch_size:]),
                history.history['loss'][0] / critic_buffer_arr.shape[0]))

            # TODO: give Logger class dictionary to keep track
            # of stuff like this. Goal: logger.logs['loss'] = history.history['loss'][0]
            # then log it in csv or json or something good with:
            # logger.log('loss')
            val_loss.append(history.history['loss'][0])
            critic_buffer = []

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
    parser.add_argument('--eps', type=int, default=100)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--save', type=str, default='')
    # use for logging
    # to-do: implement good logging procedure; can take inspiration from openai (medium)
    parser.add_argument('--exp_name', type=str, default='log1')
    args = parser.parse_args()
    print(args)
    ppo(environment=args.env, hidden_units=args.hid, gamma=args.gamma,
        seed_num=args.seed, learning_rate=args.lr, num_episodes=args.eps,
        batch_size=args.batch, num_layers=args.l, fpath=args.exp_name,
        arg_dict=vars(args), lam=args.lam, save=args.save)
