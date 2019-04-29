import gym
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from tqdm import tqdm
import time

def explore(policy, env, steps, num_trajectories, sess):
    trajectories = []
    rewards = []
    # print("\nExploring...\n")
    for i in range(num_trajectories):
        obs = np.expand_dims(env.reset(), axis = 0)
        trajectory = []
        reward_j = []
        i = 0
        for j in range(steps):
            # env.render()
            action_probs = sess.run(policy, feed_dict = {'p_obs:0': obs})
            action = np.argmax(action_probs)
            trajectory.append((obs, action))
            obs, reward, done, info = env.step(action)
            obs = np.expand_dims(obs, axis = 0)
            reward_j.append(reward)
            if done:
                break
            i += 1
        # print(action_probs)
        # print("Episode over after {0} timesteps".format(i+1))
        trajectories.append(trajectory)
        rewards.append(reward_j)
        
    return trajectories, rewards

def rtg(R, t):  #Rewards to go
    d = 1.0     # a discount factor 
    R2Go = np.copy(np.array(R[t:]))
    for i in range(len(R2Go)):
        R2Go[i] *= d**i
    rewards_to_go = np.sum(R2Go).astype(np.float64)
    return rewards_to_go

def grad_log_policy(params, action_place, observation, policy):
    W1, W2, b1, b2 = params
    with tf.name_scope("compute_gradients") as scope:
        log_policy = tf.math.log(policy)
        W1_grad = tf.squeeze(tf.gradients(tf.slice(log_policy, [0, action_place], [-1, 1]), W1))
        W2_grad = tf.squeeze(tf.gradients(tf.slice(log_policy, [0, action_place], [-1, 1]), W2))
        b1_grad = tf.squeeze(tf.gradients(tf.slice(log_policy, [0, action_place], [-1, 1]), b1))
        b2_grad = tf.squeeze(tf.gradients(tf.slice(log_policy, [0, action_place], [-1, 1]), b2))
    return (W1_grad, W2_grad, b1_grad, b2_grad)
  
  
def compute_grad(params, dims, num_trajectories, observation, policy):
    # print("Setting up gradient...")
    obs_dim, hidden_units, action_dim = dims
    with tf.name_scope('grad_vars') as scope:
        W1_grad_sum = tf.Variable(tf.zeros([obs_dim, hidden_units],    dtype = tf.float64), name = "W1_sum")
        W2_grad_sum = tf.Variable(tf.zeros([hidden_units, action_dim], dtype = tf.float64), name = "b1_sum")
        b1_grad_sum = tf.Variable(tf.zeros([hidden_units],             dtype = tf.float64), name = "W2_sum")
        b2_grad_sum = tf.Variable(tf.zeros([action_dim],               dtype = tf.float64), name = "b2_sum")
        
        N = tf.constant(num_trajectories, name = num_trajectories)
        R_place      = tf.placeholder(tf.float64, name = "R_place")
        t_place      = tf.placeholder(tf.int32, name = "t_place")
        action_place = tf.placeholder(tf.int32, name = "action_place")

    # return W1_grad_sum, W1_grad_sum, W2_grad_sum, b2_grad_sum

    W1_grad, W2_grad, b1_grad, b2_grad = grad_log_policy(
        params, action_place, observation, policy)

    with tf.name_scope('rewards_to_go') as scope:
        r2go = tf.py_func(rtg, [R_place, t_place], tf.float64)

    with tf.name_scope('estimate_policy_grad') as scope:
        W1_grad_sum = tf.add(tf.multiply(W1_grad, r2go), W1_grad_sum)
        W2_grad_sum = tf.add(tf.multiply(W2_grad, r2go), W2_grad_sum)
        b1_grad_sum = tf.add(tf.multiply(b1_grad, r2go), b1_grad_sum)
        b2_grad_sum = tf.add(tf.multiply(b2_grad, r2go), b2_grad_sum)
        grad_sums = (W1_grad_sum / N, W2_grad_sum / N, b1_grad_sum / N, b2_grad_sum / N)

    return grad_sums

def run_all(params, trajectories, rewards, sess):

    for i in range(len(trajectories)):  
        R    = rewards[i]
        Traj = trajectories[i]
        for t in range(len(Traj)):
            obs    = Traj[t][0]
            action = Traj[t][1].astype(np.int32)
            feed_dictionary = {
                "grad_vars/R_place:0": R, 
                "grad_vars/t_place:0": t,
                "p_obs:0": obs,
                "grad_vars/action_place:0": action
            }
            param_vals = sess.run(params, feed_dict = feed_dictionary)
    return param_vals

def init_mlp(dims):
    obs_dim, hidden_units, action_dim = dims
    with tf.name_scope('weights') as scope:
        initializer = xavier_initializer(dtype = tf.float64, seed = 1)
        W1 = tf.Variable(initializer([obs_dim, hidden_units]), name = 'W1')
        W2 = tf.Variable(initializer([hidden_units, action_dim]), name = 'W2')
        b1 = tf.Variable(initializer([hidden_units]), name = 'b1')
        b2 = tf.Variable(initializer([action_dim]), name = 'b2')
        params   = (W1, W2, b1, b2)
    return params
  
def mlp(func_obs, params):
    W1, W2, b1, b2 = params
    with tf.name_scope('feed_forward') as scope:
        layer_1  = tf.nn.relu(tf.add(tf.matmul(func_obs, W1), b1))
        policy = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2), b2))
    return policy

def run(epochs = 10, learning_rate = .1, 
		steps = 100, num_trajectories = 20, 
		environment = 'CartPole-v0'):

    print("Creating environment...")
    env          = gym.make(environment)
    action_dim   = env.action_space.n
    obs_dim      = len(env.observation_space.high)
    hidden_units = 10
    dims = (obs_dim, hidden_units, action_dim)  
    observation = tf.placeholder(tf.float64, [1, obs_dim], name = 'p_obs')

    print("Creating network...")
    params = init_mlp(dims)
    W1, W2, b1, b2 = params
    policy = mlp(observation, params)

    print("\nStarting session...\n")
    with tf.Session() as sess:
        print(tf.global_variables())
        lr = tf.constant(learning_rate, dtype = tf.float64, name = 'learning_rate')
        W1_grad, W2_grad, b1_grad, b2_grad = compute_grad(
            params, dims, num_trajectories, observation, policy
        )
        with tf.name_scope('update_weights') as scope:
            W1 = tf.assign(W1, tf.add(W1, lr * W1_grad))
            W2 = tf.assign(W2, tf.add(W2, lr * W2_grad))
            b1 = tf.assign(b1, tf.add(b1, lr * b1_grad))
            b2 = tf.assign(b2, tf.add(b2, lr * b2_grad))
        params = W1, W2, b1, b2
        print(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            print("Epoch:", i)
            start = time.time()
            trajectories, rewards = explore(policy, env, steps, num_trajectories, sess)
            print("Mean reward:", np.mean([len(traj) for traj in rewards]))
            # print("Time:", time.time() - start)
            # start = time.time()
            print("Computing gradients...")
            param_vals = run_all(params, trajectories, rewards, sess)
            print(param_vals)
            print("Time:", time.time() - start)
            print()
        writer = tf.summary.FileWriter("./graphs", sess.graph)
        print("\nDone!")
    return None
    env.close()

run()
