import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

# parameters
environment = 'CartPole-v0'
num_episodes = 500
batch_size = 5
hidden_units = 32
learning_rate = .01
gamma = .8
max_steps = 300
seed_num = 10

# ensure reproducibility (and make debugging possible)
np.random.seed(seed_num)
random.seed(seed_num)
os.environ['PYTHONHASHSEED'] = str(seed_num)
tf.random.set_seed(seed_num)

# set-up environment
env = gym.make(environment)
# action_dim = len(env.action_space.high)
action_dim = env.action_space.n
obs_dim = len(env.observation_space.high)

policy = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_units, input_shape = (obs_dim,), activation = 'relu'),
    tf.keras.layers.Dense(action_dim, activation = 'softmax')
    ])

optimizer = tf.keras.optimizers.Adam(lr = learning_rate)

# create buffer so we can sum gradients
grad_buffer = policy.trainable_variables
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0

rewards = []

for ep_number in range(num_episodes):
    
    ep_reward = 0
    obs = np.expand_dims(env.reset(), axis = 0)
    
    ep_buffer = []

    for j in range(max_steps):
        # env.render()
        with tf.GradientTape() as tape:
            action_probs = policy(obs)
            action = np.random.choice(action_probs.numpy()[0], p = action_probs.numpy()[0])
            action = int(np.where(action_probs.numpy()[0] == action)[0])
            log = tf.math.log(action_probs[0, action])

        grads = tape.gradient(log, policy.trainable_variables)
        obs, reward, done, info = env.step(action)
        obs = np.expand_dims(obs, axis = 0)

        ep_buffer.append([reward, grads])
        ep_reward += reward

        if done:
            break

    rewards.append(ep_reward)

    ep_buffer = np.array(ep_buffer)

    # compute rewards-to-go
    # todo: advantage function???
    rewards_to_go = np.zeros(ep_buffer.shape[0])
    for t in range(ep_buffer.shape[0]):
        length = ep_buffer.shape[0] - t
        weights = np.array([gamma ** i for i in range(length)])
        rewards_to_go[t] = np.sum(np.multiply(ep_buffer[t:,0], weights))

    # add episode information to current estimation of policy gradient
    for t, ep_info in enumerate(ep_buffer):
        for ix, grad in enumerate(ep_info[1]):
            grad_buffer[ix] -= (1 / batch_size) * grad * rewards_to_go[t]

    # every batch_size number of episodes, 
    # run gradient descent on sample
    if ep_number % batch_size == 0:

        optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))

        # re-initialize buffer to zero (on-policy)
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0

    if ep_number % 10 == 0:
        print("Episode {} reward:".format(ep_number), np.mean(rewards[-10:]))

plt.plot(rewards)
plt.show(block=False)
env.close()
print("Done!")
plt.show()