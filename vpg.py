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

def compute_grad(policy, rewards, trajectories):
    gradient = 0

    return gradient
    
def get_action(policy, observation):
	with tf.Session() as sess:
    	sess.run(tf.global_variables_initializer())
		
    
    return action

def compute_state_reward(state, action):
    #calculate reward
    return reward

def update_policy(policy, grad, learning_rate):
    #tensorflow thing to take gradient step

    return policy

def mlp(observation, seed, dims):
    obs_dim, hidden_units, action_dim = dims
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(observation)
        W1 = tf.Variable(tf.zeros([obs_dim, hidden_units]))
        W2 = tf.Variable(tf.zeros([hidden_units, action_dim]))
        b1 = tf.Variable(tf.zeros([hidden_units]))
        b2 = tf.Variable(tf.zeros([action_dim]))
        layer_1 = tf.nn.relu(tf.add(tf.matmul(observation, W1), b1))
        layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2)))
        log_out = tf.math.log(layer_out)
    return layer_out, tape

def cost_function(policy):
    cost = 0

    return cost


def run(epochs = 5, learning_rate = .01, seed = 1, steps = 20, num_episodes = 100, environment = 'CartPole-v0'):
    env = gym.make(environment)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.n
    hidden_units = 10
    dims = (env.observation_space.n, hidden_units, env.action_space.n)

    observation = tf.placeholder(tf.float32, [1, dims])
    policy, tape = mlp(observation, seed, action_dim, obs_dim, hidden_units)

    for i in range(epochs):
        trajectories, rewards = explore(policy, env, steps, num_episodes)
        grad = compute_grad(trajectories, rewards, policy)
        policy = update_policy(policy, grad, learning_rate)
    
    print("Done!")
    
    env.close()




