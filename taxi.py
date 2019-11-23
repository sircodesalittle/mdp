import numpy as np
import gym

from agents import ValueIterationAgent, PolicyIterationAgent, QLearningAgent

gamma = .1


# Value Iteration
#######################################################
env = gym.make('Taxi-v3')
agent = ValueIterationAgent(env, gamma)
print('Average reward: ' + str(np.mean(agent.scores)))


# Policy Iteration
#######################################################
env = gym.make('Taxi-v3')
agent = PolicyIterationAgent(env, gamma)
print('Average reward: ' + str(np.mean(agent.scores)))


# Q Learning
#######################################################
env = gym.make('Taxi-v3')
agent = QLearningAgent(env)
q, stats = agent.q_learning(env, 500)
