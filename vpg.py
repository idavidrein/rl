import gym
import tensorflow as tf
import numpy as np

def explore(policy, env, steps, num_episodes):
    trajectories = []
    rewards = []
    for i in range(num_episodes):
        observation = env.reset()
        trajectory = []
        reward_i = []
        for i in range(steps):
            action = get_action(policy, observation)
            trajectory.append((observation, action))
            observation, reward, done, info = env.step(action)
            reward_i.append(reward)
            if done:
                print("Episode over after {0} timesteps".format(i+1))
                break
        trajectories.append(trajectory)
        rewards.append(reward_i)
        
    return trajectories, rewards

def compute_grad(trajectories, rewards, policy):
    gradient = 0

    return gradient
    
def get_action(policy, observation):
    action = 0
    # feedforward through network
    
    return action

def compute_state_reward(state, action):
    #calculate reward
    return reward

def update_policy(policy, grad, learning_rate):
    #tensorflow thing to take gradient step

    return policy

def init_policy(seed, action_dim, obs_dim):
    hidden_units = 10
    observation = tf.placeholder(tf.float21, [None, obs_dim])
    W1 = tf.Variable(tf.zeros([obs_dim, hidden_units]))
    W2 = tf.Variable(tf.zeros([hidden_units, action_dim]))
    b1 = tf.Variable(tf.zeros([hidden_units]))
    b2 = tf.Variable(tf.zeros([action_dim]))
    return policy

def cost_function(policy):
    cost = 0

    return cost


def run(epochs = 5, learning_rate = .01, seed = 1, steps = 20, num_episodes = 100, environment = 'CartPole-v0'):
    env = gym.make(environment)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.n
    policy = init_policy(seed, action_dim, obs_dim)
    for i in range(epochs):
        trajectories, rewards = explore(policy, env, steps, num_episodes)
        grad = compute_grad(trajectories, rewards, policy)
        policy = update_policy(policy, grad, learning_rate)
    
    print("Done!")
    
    env.close()




