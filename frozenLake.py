import numpy as np
import gym

import plotting
from agents import ValueIterationAgent, PolicyIterationAgent, QLearningAgent

gamma = .1


# Value Iteration
#######################################################
env = gym.make('FrozenLake-v0')
agent = ValueIterationAgent(env, gamma)
print('Average reward: ' + str(np.mean(agent.scores)))


# Policy Iteration
#######################################################
env = gym.make('FrozenLake-v0')
agent = PolicyIterationAgent(env, gamma)
print('Average reward: ' + str(np.mean(agent.scores)))


# Q Learning
#######################################################
env = gym.make('FrozenLake-v0')
agent = QLearningAgent(env)
Q, stats = agent.q_learning(env, 500)

plotting.plot_episode_stats(stats)
