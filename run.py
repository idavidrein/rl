import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import gym
from gym.spaces import Discrete


MAX_STEPS = 1000
ENVIRONMENT = "CartPole-v0"

env = gym.make(ENVIRONMENT)
policy = load_model("models/CartPole_policy")

while True:

    obs = np.array(env.reset()).reshape(1, -1)
    num_steps = 0

    for i in range(MAX_STEPS):
        num_steps += 1
        env.render()
        if isinstance(env.action_space, Discrete):
            action_probs = policy(obs)
            log_probs = tf.math.log(action_probs)
            action = int(tf.random.categorical(log_probs, 1))
        else:
            action, log = policy(obs)
            action = np.squeeze(action)

        obs, reward, done, info = env.step(action)
        obs = np.array(obs).reshape(1, -1)

        if done:
            break

    print(num_steps, end='\r')