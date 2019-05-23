#Reinforcement Learning (RL)
Implementations of common Reinforcement Learning Algorithms in Tensorflow 2.0. Uses the OpenAI Gym environment for algorithm testing.

## Algorithms Implemented

* Vanilla Policy Gradient: Similar to the REINFORCE algorithm, with on-policy Monte Carlo sampling. Uses Adam for optimization. 

* A2C: Modification of VPG where a value function is learned to estimate discounted rewards, and Generalized Advantage Estimation is used to estimate the advantage function. Policy updates are biased, but with lower variance than with VPG.
