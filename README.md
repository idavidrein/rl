# Reinforcement Learning (RL)
Implementations of common Reinforcement Learning Algorithms in Tensorflow 2.0. Uses the OpenAI Gym environment for algorithm testing.

## Algorithms Implemented

* Vanilla Policy Gradient (VPG): Similar to the REINFORCE algorithm, with on-policy Monte Carlo sampling. Uses Adam for optimization. 

* Advantage Actor-Critic (A2C): Modification of VPG where a value function is learned to estimate discounted rewards, and Generalized Advantage Estimation is used to estimate the advantage function. Policy updates are biased, but with lower variance than with VPG.

* Proximal Policy Optimization (PPO-Clip): An A2C algorithm in which a clipped surrogate objective function is used to discourage large (and likely harmful) policy updates.