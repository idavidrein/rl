import gym
import tensorflow as tf


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

def update_policy(policy, grad):
    #tensorflow thing to take gradient step

    return policy

def init_policy(seed, num_actions = 2):
    policy = tf.keras.models.Sequential()
    width = 32
    policy.add(tf.keras.layers.Dense(width, activation = tf.nn.relu))  
    policy.add(tf.keras.layers.Dense(num_actions, activation = tf.nn.softmax))
    
    return policy

def cost_function(policy):
    cost = 0

    return cost


def run(epochs = 5, seed = 1, steps = 20, num_episodes = 100, environment = 'CartPole-v0'):
    env = gym.make(environment)
    policy = init_policy(seed, num_actions)
    for i in range(epochs):
        trajectories, rewards = explore(policy, env, steps, num_episodes)
        grad   = compute_grad(trajectories, rewards, policy)
        policy = update_policy(policy, grad)
    
    print("Done!")
    
    env.close()




