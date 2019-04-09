import gym
import tensorflow as tf


def explore(policy):
	trajectories = {}

	return trajectories

def compute_grad(trajectories, rewards, policy):
	gradient = 0

	return gradient

def compute_rewards(trajectories):
	rewards = {}
	for trajectory in trajectories:
		#sum compute_state_reward

	return rewards

def compute_state_reward(state, action):
	#calculate reward
	return reward

def update_policy(policy, grad):
	#update params via backpropagation

	return policy

def init_policy(seed):
	policy = 3

	return policy

def cost_function(policy):
	cost = 0

	return cost


def run(epochs = 5, seed = 1):
	policy = init_policy(seed)

	for i in range(epochs):
		trajectories = explore(policy)
		rewards = compute_rewards(trajectories)
		grad = compute_grad(trajectories, rewards, policy)
		policy = update_policy(policy, grad)
		

