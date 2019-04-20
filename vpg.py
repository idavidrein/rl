import gym
import tensorflow as tf
import numpy as np

def explore(policy, env, steps, num_episodes, sess):
    trajectories = []
    rewards = []
    for i in range(num_episodes):
        observation = env.reset()
        trajectory = []
        reward_i = []
        for i in range(steps):
            action = sess.run([policy], observation)
            trajectory.append((observation, action))
            observation, reward, done, info = env.step(action)
            reward_i.append(reward)
            if done:
                print("Episode over after {0} timesteps".format(i+1))
                break
        trajectories.append(trajectory)
        rewards.append(reward_i)
        
    return trajectories, rewards

def rtg(R, t):  #Rewards to go
    d = 1.0                         # a discount factor 
    R2Go = np.copy(np.array(R[t:]))
    for i in range(len(R2Go)):
        R2Go[i] *= d**i
    SamsAwesomeValue = np.sum(R2Go)
    return(SamsAwesomeValue)

def grad_log_policy(params, action, obs, sess):
    W1, W2, b1, b2 = params
    with tf.GradientTape(persistant = True) as tape:
      tape.watch(params)
      layer_out = mlp(obs, params)
      log_out = tf.math.log(layer_out)
    W1_grad = tape.gradient(log_out, W1)
    W2_grad = tape.gradient(log_out, W2)
    b1_grad = tape.gradient(log_out, b1)
    b2_grad = tape.gradient(log_out, b2)
    return (W1_grad, W2_grad, b1_grad, b2_grad)
  
  
def compute_grad(params, rewards, trajectories, dims, sess):
    obs_dim, hidden_units, action_dim = dims
    W1_grad_sum = tf.Variable(tf.zeros([obs_dim, hidden_units]))
    W2_grad_sum = tf.Variable(tf.zeros([hidden_units, action_dim]))
    b1_grad_sum = tf.Variable(tf.zeros([hidden_units]))
    b2_grad_sum = tf.Variable(tf.zeros([action_dim]))
    
    N = len(trajectories)
    
    for i in range(len(trajectories)):  
        R    = rewards[i]
        Traj = trajectories[i]
        for t in range(len(Traj)):
            obs    = Traj[t][0]
            action = Traj[t][1]
            (W1_grad, W2_grad, b1_grad, b2_grad) = grad_log_policy(params, action, obs, sess) * rtg(R, t)
            W1_grad_sum = tf.add(W1_grad, W1_grad_sum)
			W2_grad_sum = tf.add(W2_grad, W2_grad_sum)
			b1_grad_sum = tf.add(b1_grad, b1_grad_sum)
			b2_grad_sum = tf.add(b2_grad, b2_grad_sum)
  
    return ( W1_grad_sum/N , W2_grad_sum/N , b1_grad_sum/N , b2_grad_sum/N ) 

def init_mlp(dims):
    obs_dim, hidden_units, action_dim = dims
    W1 = tf.Variable(tf.zeros([obs_dim, hidden_units]))
    W2 = tf.Variable(tf.zeros([hidden_units, action_dim]))
    b1 = tf.Variable(tf.zeros([hidden_units]))
    b2 = tf.Variable(tf.zeros([action_dim]))
    params   = (W1, W2, b1, b2)
    return params
  
def mlp(observation, params):
  	W1, W2, b1, b2 = params
    layer_1  = tf.nn.relu(tf.add(tf.matmul(observation, W1), b1))
    layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2), b2)) #Minor Change: Sam added the b2 here
    return layer_out

def run(epochs = 5, learning_rate = .01, steps = 20, num_episodes = 100, environment = 'CartPole-v0'):
    env          = gym.make(environment)
    action_dim   = env.action_space.n
    obs_dim      = env.observation_space.n
    hidden_units = 10
    dims = (obs_dim, hidden_units, action_dim) #Minor Change: Sam changed env.observation_space.n to obs_dim, etc. 

    observation = tf.placeholder(tf.float32, [1, dims])
    params = init_mlp(dims)
  	layer_out = mlp(observation, params)

    with tf.Session() as sess:
	    for i in range(epochs):
	        trajectories, rewards = explore(layer_out, env, steps, num_episodes, sess)
            
	        W1_grad, W2_grad, b1_grad, b2_grad  = compute_grad(params, rewards, trajectories, dims, sess)
            W1_new = W1.assign(tf.add(W1, learning_rate * W1_grad))
	        W2_new = W2.assign(tf.add(W2, learning_rate * W2_grad))
            b1_new = b1.assign(tf.add(b1, learning_rate * b1_grad))
            b2_new = b2.assign(tf.add(b2, learning_rate * b2_grad))
    
    
    print("Done!")
    
    env.close()
