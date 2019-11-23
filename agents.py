import itertools
import numpy as np
import sys

from collections import defaultdict
import plotting

MAX_MAX_ITERATIONS = 10000
RENDER = False


class ValueIterationAgent:
    MAX_ITERATIONS = MAX_MAX_ITERATIONS

    def __init__(self, env, gamma):
        self.gamma = gamma
        optimal_v, self.convergence = ValueIterationAgent.value_iteration(env)
        self.policy = ValueIterationAgent.extract_policy(env, optimal_v, gamma)
        self.scores = ValueIterationAgent.evaluate_policy(env, self.policy)

    @staticmethod
    def run_episode(env, policy, gamma=1.0, render=False):
        """ Evaluates policy by using it to run an episode and finding its
        total reward.
        args:
        env: gym environment.
        policy: the policy to be used.
        gamma: discount factor.
        render: boolean to turn rendering on/off.
        returns:
        total reward: real value of the total reward recieved by agent under policy.
        """
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

    @staticmethod
    def evaluate_policy(env, policy, gamma=1.0, n=100):
        """ Evaluates a policy by running it n times.
        returns:
        average total reward
        """
        scores = [
            ValueIterationAgent.run_episode(env, policy, gamma=gamma, render=RENDER)
            for _ in range(n)]
        return np.mean(scores)

    @staticmethod
    def extract_policy(env, v, gamma=1.0):
        """ Extract the policy given a value-function """
        policy = np.zeros(env.nS)
        for s in range(env.nS):
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for next_sr in env.P[s][a]:
                    # next_sr is a tuple of (probability, next state, reward, done)
                    p, s_, r, _ = next_sr
                    q_sa[a] += (p * (r + gamma * v[s_]))
            policy[s] = np.argmax(q_sa)
        return policy

    @staticmethod
    def value_iteration(env):
        """ Value-iteration algorithm """
        v = np.zeros(env.nS)  # initialize value-function
        eps = 1e-10
        convergence = ValueIterationAgent.MAX_ITERATIONS
        for i in range(ValueIterationAgent.MAX_ITERATIONS):
            prev_v = np.copy(v)
            for s in range(env.nS):
                q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
                v[s] = max(q_sa)
            if np.sum(np.fabs(prev_v - v)) <= eps:
                convergence = i + 1
                break
        return v, convergence


class PolicyIterationAgent:
    MAX_ITERATIONS = MAX_MAX_ITERATIONS

    def __init__(self, env, gamma):
        self.gamma = gamma
        self.policy, self.convergence = PolicyIterationAgent.policy_iteration(env, gamma)
        self.scores = PolicyIterationAgent.evaluate_policy(env, self.policy)

    @staticmethod
    def run_episode(env, policy, gamma=1.0, render=False):
        """ Runs an episode and return the total reward """
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done:
                break
        return total_reward

    @staticmethod
    def evaluate_policy(env, policy, gamma=1.0, n=1000):
        scores = [PolicyIterationAgent.run_episode(env, policy, gamma, RENDER) for _ in range(n)]
        return np.mean(scores)

    @staticmethod
    def extract_policy(env, v, gamma=1.0):
        """ Extract the policy given a value-function """
        policy = np.zeros(env.nS)
        for s in range(env.nS):
            q_sa = np.zeros(env.nA)
            for a in range(env.nA):
                q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
            policy[s] = np.argmax(q_sa)
        return policy

    @staticmethod
    def compute_policy_v(env, policy, gamma=1.0):
        """ Iteratively evaluate the value-function under policy.
        Alternatively, we could formulate a set of linear equations in iterms of v[s]
        and solve them to find the value function.
        """
        v = np.zeros(env.nS)
        eps = 1e-10
        while True:
            prev_v = np.copy(v)
            for s in range(env.nS):
                policy_a = policy[s]
                v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
            if np.sum((np.fabs(prev_v - v))) <= eps:
                # value converged
                break
        return v

    @staticmethod
    def policy_iteration(env, gamma):
        """ Policy-Iteration algorithm """
        policy = np.random.choice(env.nA, size=env.nS)  # initialize a random policy
        convergence_step = 0
        for i in range(PolicyIterationAgent.MAX_ITERATIONS):
            old_policy_v = PolicyIterationAgent.compute_policy_v(env, policy, gamma)
            new_policy = PolicyIterationAgent.extract_policy(env, old_policy_v, gamma)
            if np.all(policy == new_policy):
                convergence_step = i + 1
                break
            policy = new_policy
        return policy, convergence_step


class QLearningAgent:

    def __init__(self, env):
        self.env = env

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def q_learning(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: OpenAI environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        policy = self.make_epsilon_greedy_policy(Q, epsilon, self.env.action_space.n)

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                # print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # Reset the environment and pick the first action
            state = self.env.reset()

            # One step in the environment
            # total_reward = 0.0
            for t in itertools.count():

                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                if done:
                    break

                state = next_state

        return Q, stats
