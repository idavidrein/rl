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

def rtg(R, t):  #rewards to go
    d = 1.0                      # a discount factor 
    R2Go = np.copy(np.array(R[t:])) # Copy the Rewards
    for i in range(len(R2Go)):  # 
        R2Go[i] *= d**i
    SamsAwesomeValue = np.sum(R2Go)
    return(SamsAwesomeValue)

def grad_log_policy(policy, action, obs, sess):
    #need to take gradient[log(policy[action | obs])] 
    return(0)

def compute_grad(policy, rewards, trajectories, sess):
    grad_sum = 0
    for i in range(len(trajectories)):  
        R    = rewards[i]
        Traj = trajectories[i]
        for t in range(len(Traj)):
            obs    = Traj[t][0]
            action = Traj[t][1]
            grad_sum += grad_log_policy(policy, action, obs, sess) * rtg(R, t)
    gradient = grad_sum/len(trajectories)
    # I'm guessing this gradient needs to be a tuple of size four for W1, W2, b1, b2
    return gradient

def mlp(observation, seed, dims):
    obs_dim, hidden_units, action_dim = dims
    W1 = tf.Variable(tf.zeros([obs_dim, hidden_units]))
    W2 = tf.Variable(tf.zeros([hidden_units, action_dim]))
    b1 = tf.Variable(tf.zeros([hidden_units]))
    b2 = tf.Variable(tf.zeros([action_dim]))
    params   = (W1, W2, b1, b2)
    layer_1  = tf.nn.relu(tf.add(tf.matmul(observation, W1), b1))
    layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_1, W2), b2)) #Minor Change: Sam added the b2 here
    log_out   = tf.math.log(layer_out)
    return layer_out, log_out, params

def run(epochs = 5, learning_rate = .01, seed = 1, steps = 20, num_episodes = 100, environment = 'CartPole-v0'):
    env          = gym.make(environment)
    action_dim   = env.action_space.n
    obs_dim      = env.observation_space.n
    hidden_units = 10
    dims = (obs_dim, hidden_units, action_dim) #Minor Change: Sam changed env.observation_space.n to obs_dim, etc. 

    observation = tf.placeholder(tf.float32, [1, dims])
    policy, log_policy, params = mlp(observation, seed, (obs_dim, hidden_units, action_dim))

    with tf.Session() as sess:
	    for i in range(epochs):
	        trajectories, rewards = explore(policy, env, steps, num_episodes, sess)
	        grad   = compute_grad(policy, rewards, trajectories, sess)
	        policy = tf.add(policy, learning_rate * grad) #Don't we need to like update the params 
                                                          # and then load them into the policy?
    
    print("Done!")
    
    env.close()


















