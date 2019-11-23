import numpy as np
import gym

import plotting
from agents import ValueIterationAgent, PolicyIterationAgent, QLearningAgent


class Experiment:

    def __init__(self, name, env_name):
        print('Beginning Experiment: ' + name)
        self.env = gym.make(env_name)
        self.gamma = .1

    def run(self):
        print('\tValue Iteration')
        vi_agent = ValueIterationAgent(self.env, self.gamma)
        print('\t\tAverage reward: ' + str(np.mean(vi_agent.scores)))
        print('\t\tConvergence step: ' + str(vi_agent.convergence))
        print('\t\tPolicy: ' + str(vi_agent.policy))

        print('\tPolicy Iteration')
        self.env.reset()
        pi_agent = PolicyIterationAgent(self.env, self.gamma)
        print('\t\tAverage reward: ' + str(np.mean(pi_agent.scores)))
        print('\t\tConvergence step: ' + str(pi_agent.convergence))
        print('\t\tPolicy: ' + str(pi_agent.policy))

        print('\tQ Learning')
        self.env.reset()
        ql_agent = QLearningAgent(self.env)
        q, stats = ql_agent.q_learning(self.env, 500)
        plotting.plot_episode_stats(stats)


frozenLakeExperiment = Experiment('Frozen Lake', 'FrozenLake-v0')
frozenLakeExperiment.run()

taxiExperiment = Experiment('Taxi', 'Taxi-v3')
taxiExperiment.run()
