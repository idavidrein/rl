import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def run(learning_rate = .1, batch_size = 20,
    max_steps = 200, num_episodes = 100, 
    environment = 'CartPole-v0'):

    env          = gym.make(environment)
    action_dim   = env.action_space.n
    obs_dim      = len(env.observation_space.high)
    hidden_units = 32
    dims = (obs_dim, hidden_units, action_dim)  

    policy = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape = (4,), activation = 'relu'),
        tf.keras.layers.Dense(2, activation = 'softmax')
        ])

    optimizer = tf.keras.optimizers.Adam()

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
                action = int(np.random.uniform() >= action_probs.numpy()[0,1])
                log = tf.math.log(action_probs[obs])

            grads = tape.gradient(log, policy.trainable_variables)
            
            obs, reward, done, info = env.step(action)
            obs = np.expand_dims(obs, axis = 0)

            ep_buffer.append([reward, grads])
            ep_reward += reward

            if done:
                break

        rewards.append(ep_reward)
        if ep_number % batch_size == 0:
            print("Episode {} reward:".format(ep_number), ep_reward)

        ep_buffer = np.array(ep_buffer)

        # compute rewards-to-go
        # todo: include discounting (and advantage function too!)
        rewards_to_go = np.zeros(ep_buffer.shape[0])
        for t in range(ep_buffer.shape[0]):
            rewards_to_go[t] = np.sum(ep_buffer[t:,0])

        for t, ep_info in enumerate(ep_buffer):
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] += (1 / batch_size) * ep_info[1][ix] * rewards_to_go[t]

        if ep_number % batch_size == 0:
            optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0


    plt.plot(rewards)
    plt.show()
    env.close()
    return None

run()
