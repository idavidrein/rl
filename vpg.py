import gym
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from tqdm import tqdm

def explore(layer_out, env, steps, num_trajectories, sess):
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
            action_probs = sess.run(layer_out, feed_dict = {'p_obs:0': obs})
            action = np.argmax(action_probs)
            trajectory.append((obs, action))
            obs, reward, done, info = env.step(action)
            obs = np.expand_dims(obs, axis = 0)
            reward_j.append(reward)
            if done:
                break
            i += 1
        # print("Episode over after {0} timesteps".format(i+1))
        trajectories.append(trajectory)
        rewards.append(reward_j)
        
    return trajectories, rewards

def rtg(R, t):  #Rewards to go
    d = 1.0     # a discount factor 
    R2Go = np.copy(np.array(R[t:]))
    for i in range(len(R2Go)):
        R2Go[i] *= d**i
    SamsAwesomeValue = np.sum(R2Go).astype(np.float64)
    return(SamsAwesomeValue)

def grad_log_policy(params, action_place, obs_place, sess):
    W1, W2, b1, b2 = params
    layer_out = mlp(obs_place, params)
    log_out = tf.math.log(layer_out)
    W1_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action_place], [-1, 1]), W1))
    W2_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action_place], [-1, 1]), W2))
    b1_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action_place], [-1, 1]), b1))
    b2_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action_place], [-1, 1]), b2))
    return (W1_grad, W2_grad, b1_grad, b2_grad)
  
  
def compute_grad(params, rewards, trajectories, dims, sess):
    # print("Setting up gradient...")
    obs_dim, hidden_units, action_dim = dims
    W1_grad_sum = tf.Variable(tf.zeros([obs_dim, hidden_units], dtype = tf.float64))
    W2_grad_sum = tf.Variable(tf.zeros([hidden_units, action_dim], dtype = tf.float64))
    b1_grad_sum = tf.Variable(tf.zeros([hidden_units], dtype = tf.float64))
    b2_grad_sum = tf.Variable(tf.zeros([action_dim], dtype = tf.float64))
    sess.run(tf.global_variables_initializer())
    
    N = len(trajectories)
    R_place      = tf.placeholder(tf.float64, name = "R_place")
    t_place      = tf.placeholder(tf.int32, name = "t_place")
    obs_place    = tf.placeholder(tf.float64, shape = [1, obs_dim], name = "obs_place")
    action_place = tf.placeholder(tf.int32, name = "action_place")

    W1_grad, W2_grad, b1_grad, b2_grad = grad_log_policy(
        params, action_place, obs_place, sess
    )

    asdf = tf.py_function(rtg, [R_place, t_place], tf.float64)

    W1_grad_sum = tf.add(tf.multiply( W1_grad, asdf), W1_grad_sum )
    W2_grad_sum = tf.add(tf.multiply( W2_grad, asdf), W2_grad_sum )
    b1_grad_sum = tf.add(tf.multiply( b1_grad, asdf), b1_grad_sum )
    b2_grad_sum = tf.add(tf.multiply( b2_grad, asdf), b2_grad_sum )

    for i in range(len(trajectories)):  
        R    = rewards[i]
        Traj = trajectories[i]
        for t in range(len(Traj)):
            obs    = Traj[t][0]
            action = Traj[t][1].astype(np.int32)
            feed_dictionary = {
                R_place: R, 
                t_place: t,
                obs_place: obs,
                action_place: action
            }
            param_vals = sess.run(
                [W1_grad_sum, W2_grad_sum, b1_grad_sum, b2_grad_sum],
                feed_dict = feed_dictionary)

    W1_grad_sum_val, W2_grad_sum_val, b1_grad_sum_val, b2_grad_sum_val = param_vals
    return ( W1_grad_sum_val/N , W2_grad_sum_val/N , b1_grad_sum_val/N , b2_grad_sum_val/N ) 

def init_mlp(dims):
    obs_dim, hidden_units, action_dim = dims
    initializer = xavier_initializer(dtype = tf.float64)
    W1 = tf.Variable(initializer([obs_dim, hidden_units]), name = 'W1')
    W2 = tf.Variable(initializer([hidden_units, action_dim]), name = 'W2')
    b1 = tf.Variable(initializer([hidden_units]), name = 'b1')
    b2 = tf.Variable(initializer([action_dim]), name = 'b2')
    params   = (W1, W2, b1, b2)
    return params
  
def mlp(func_obs, params):
    W1, W2, b1, b2 = params
    layer_1  = tf.nn.relu(tf.add(tf.matmul(func_obs, W1), b1))
    layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2), b2)) #Minor Change: Sam added the b2 here
    return layer_out

def run(epochs = 30, learning_rate = .1, 
		steps = 40, num_trajectories = 100, 
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
    layer_out = mlp(observation, params)
    print("\nStarting session...\n")
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter("./graphs", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(epochs)):
            trajectories, rewards = explore(layer_out, env, steps, num_trajectories, sess)
            W1_grad, W2_grad, b1_grad, b2_grad = compute_grad(params, rewards, trajectories, dims, sess)
            W1 = tf.assign(W1, tf.add(W1, learning_rate * W1_grad))
            W2 = tf.assign(W2, tf.add(W2, learning_rate * W2_grad))
            b1 = tf.assign(b1, tf.add(b1, learning_rate * b1_grad))
            b2 = tf.assign(b2, tf.add(b2, learning_rate * b2_grad))
            params = W1, W2, b1, b2
            # print("Computing gradients...")
            qwer = sess.run(params)
            layer_out = mlp(observation, params)
        # writer = tf.summary.FileWriter("./graphs", sess.graph)
        print("\nDone!")
    return None
    env.close()

run()
