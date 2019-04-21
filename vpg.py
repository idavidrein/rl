import gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def explore(layer_out, env, steps, num_episodes, sess):
    trajectories = []
    rewards = []
    print("Exploring\n")
    for i in range(num_episodes):
        obs = np.expand_dims(env.reset(), axis = 0)
        trajectory = []
        reward_i = []
        for i in range(steps):
            action_probs = sess.run(layer_out, feed_dict = {'p_obs:0': obs})
            action = np.argmax(action_probs)
            trajectory.append((obs, action))
            obs, reward, done, info = env.step(action)
            obs = np.expand_dims(obs, axis = 0)
            reward_i.append(reward)
            if done:
                print("Episode over after {0} timesteps".format(i+1))
                break
        trajectories.append(trajectory)
        rewards.append(reward_i)
        
    return trajectories, rewards

def rtg(R, t):  #Rewards to go
    d = 1.0     # a discount factor 
    R2Go = np.copy(np.array(R[t:]))
    for i in range(len(R2Go)):
        R2Go[i] *= d**i
    SamsAwesomeValue = np.sum(R2Go).astype(np.float64)
    return(SamsAwesomeValue)

def grad_log_policy(params, action, obs, sess):
    W1, W2, b1, b2 = params
    layer_out = mlp(obs, params)
    log_out = tf.math.log(layer_out)
    W1_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action], [-1, 1]), W1))
    W2_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action], [-1, 1]), W2))
    b1_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action], [-1, 1]), b1))
    b2_grad = tf.squeeze(tf.gradients(tf.slice(log_out, [0, action], [-1, 1]), b2))
    return (W1_grad, W2_grad, b1_grad, b2_grad)
  
  
def compute_grad(params, rewards, trajectories, dims, sess):
    print("Setting up gradient")
    obs_dim, hidden_units, action_dim = dims
    W1_grad_sum = tf.Variable(tf.zeros([obs_dim, hidden_units], dtype = tf.float64))
    W2_grad_sum = tf.Variable(tf.zeros([hidden_units, action_dim], dtype = tf.float64))
    b1_grad_sum = tf.Variable(tf.zeros([hidden_units], dtype = tf.float64))
    b2_grad_sum = tf.Variable(tf.zeros([action_dim], dtype = tf.float64))
    sess.run(tf.global_variables_initializer())
    
    N = len(trajectories)
    
    for i in tqdm(range(len(trajectories))):  
        R    = rewards[i]
        Traj = trajectories[i]
        for t in range(len(Traj)):
            obs    = Traj[t][0]
            action = Traj[t][1]
            W1_grad, W2_grad, b1_grad, b2_grad = grad_log_policy(params, action, obs, sess)

            W1_grad_sum = tf.add(tf.multiply( W1_grad, rtg(R,t)), W1_grad_sum )
            W2_grad_sum = tf.add(tf.multiply( W2_grad, rtg(R,t)), W2_grad_sum )
            b1_grad_sum = tf.add(tf.multiply( b1_grad, rtg(R,t)), b1_grad_sum )
            b2_grad_sum = tf.add(tf.multiply( b2_grad, rtg(R,t)), b2_grad_sum )
  
    return ( W1_grad_sum/N , W2_grad_sum/N , b1_grad_sum/N , b2_grad_sum/N ) 

def init_mlp(dims):
    obs_dim, hidden_units, action_dim = dims
    # to-do: use Xavier (or something else) to initialize instead of zeros
    initializer = tf.contrib.layers.xavier_initializer()
    W1 = tf.Variable(tf.zeros([obs_dim, hidden_units],    dtype = tf.float64), name = 'W1')
    W2 = tf.Variable(tf.zeros([hidden_units, action_dim], dtype = tf.float64), name = 'W2')
    b1 = tf.Variable(tf.zeros([hidden_units],             dtype = tf.float64), name = 'b1')
    b2 = tf.Variable(tf.zeros([action_dim],               dtype = tf.float64), name = 'b2')
    params   = (W1, W2, b1, b2)
    return params
  
def mlp(func_obs, params):
    W1, W2, b1, b2 = params
    layer_1  = tf.nn.relu(tf.add(tf.matmul(func_obs, W1), b1))
    layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2), b2)) #Minor Change: Sam added the b2 here
    return layer_out

def run(epochs = 5, learning_rate = .01, 
		steps = 20, num_episodes = 2, 
		environment = 'CartPole-v0'):
    print("running vpg\n")
    print("creating environment\n")
    env          = gym.make(environment)
    action_dim   = env.action_space.n
    obs_dim      = len(env.observation_space.high)
    hidden_units = 10
    dims = (obs_dim, hidden_units, action_dim) #Minor Change: Sam changed env.observation_space.n to obs_dim, etc. 
    print("placeholders \n")
    observation = tf.placeholder(tf.float64, [1, obs_dim], name = 'p_obs')
    print("initializing mlp \n")
    params = init_mlp(dims)
    W1, W2, b1, b2 = params
    print("creating mlp\n")
    layer_out = mlp(observation, params)
    print("starting session\n")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            trajectories, rewards = explore(layer_out, env, steps, num_episodes, sess)

            W1_grad, W2_grad, b1_grad, b2_grad = compute_grad(params, rewards, trajectories, dims, sess)
            print(W1_grad) # Tensor("truediv:0", shape=(1, 4, 10), dtype=float64)
            W1_new = W1.assign(tf.add(W1, learning_rate * W1_grad))
            W2_new = W2.assign(tf.add(W2, learning_rate * W2_grad))
            b1_new = b1.assign(tf.add(b1, learning_rate * b1_grad))
            b2_new = b2.assign(tf.add(b2, learning_rate * b2_grad))
            params = W1_new, W2_new, b1_new, b2_new
            params = sess.run(params)
            layer_out = mlp(observation, params)
            print(params)
            break
            # at this point, the updates haven't been computed yet. 
            # the way it's currently set up, they'll be computed in line
            # action_probs = sess.run(layer_out, feed_dict = {'p_obs:0': obs})
            # and a bunch of stuff (including all of the gradients) will be computed
    	print("Done!")
        return(layer_out)
    env.close()
