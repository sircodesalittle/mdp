import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, smoothing_window=10, noshow=False, experiment_name=''):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time - " + experiment_name)
    if noshow:
        plt.close(fig1)
    else:
        plt.savefig('charts/' + experiment_name + '_eps9_qlearn_episode_len_time.png')
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window) + ' - ' + experiment_name)
    if noshow:
        plt.close(fig2)
    else:
        plt.savefig('charts/' + experiment_name + '_eps9_qlearn_episode_reward_time.png')
        plt.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step - " + experiment_name)
    if noshow:
        plt.close()
    else:
        plt.savefig('charts/' + experiment_name + '_eps9_qlearn_episode_time_step.png')
        plt.show()

    return fig1, fig2, fig3


def epsilon_graph():
    """
    Epsilon 1e-1
    Beginning Experiment: Frozen Lake
        Value Iteration
            Average reward: 0.78
            Convergence step: 26
            Time: 0.0136
        Policy Iteration
            Average reward: 0.0
            Convergence step: 2
            Time: 0.00483
    Beginning Experiment: Taxi
        Value Iteration
            Average reward: -193.69
            Convergence step: 10000
            Time: 157.1
        Policy Iteration
            Average reward: -187.003
            Convergence step: 8
            Time: 0.265

    Epsilon 1e-5
    Beginning Experiment: Frozen Lake
        Value Iteration
            Average reward: 0.68
            Convergence step: 408
            Time: 0.146
        Policy Iteration
            Average reward: 0.456
            Convergence step: 3
            Time: 0.0247
    Beginning Experiment: Taxi
        Value Iteration
            Average reward: -200
            Convergence step: 10000
            Time: 125.62
        Policy Iteration
            Average reward: -123.355
            Convergence step: 11
            Time: 0.629

    Epsilon 1e-10
    Beginning Experiment: Frozen Lake
        Value Iteration
            Average reward: 0.77
            Convergence step: 877
            Time: 0.506
        Policy Iteration
            Average reward: 0.419
            Convergence step: 4
            Time: 0.00595
    Beginning Experiment: Taxi
        Value Iteration
            Average reward: -200
            Convergence step: 10000
            Time: 143.22
        Policy Iteration
            Average reward: -0.408
            Convergence step: 15
            Time: 0.740

    Epsilon 1e-15
    Beginning Experiment: Frozen Lake
        Value Iteration
            Average reward: 0.77
            Convergence step: 1333
            Time: 1.11
        Policy Iteration
            Average reward: 0.425
            Convergence step: 4
            Time: 0.00737
    Beginning Experiment: Taxi
        Value Iteration
            Average reward: -193.68
            Convergence step: 10000
            Time: 152.11
        Policy Iteration
            Average reward: 7.955
            Convergence step: 15
            Time: 1.143
    """
    epsilons = [1e-15, 1e-10, 1e-5, 1e-1]
    fl_vi_reward = [.78, .68, .77, .77]
    fl_vi_c_step = [26, 408, 877, 1333]
    fl_vi_c_time = [.0136, .146, .506, 1.11]
    fl_pi_reward = [0, .456, .419, .425]
    fl_pi_c_step = [2, 3, 4, 4]
    fl_pi_c_time = [.00483, .0247, .00595, .00737]
    tax_vi_reward = [-193.69, -200, -200, -193.68]
    tax_vi_c_step = [10000, 10000, 10000, 10000]
    tax_vi_c_time = [157.1, 125, 143, 152]
    tax_pi_reward = [-187, -123, -.408, 7.995]
    tax_pi_c_step = [8, 11, 15, 15]
    tax_pi_c_time = [.265, .629, .740, 1.143]

    plt.title('Frozen Lake Reward and Epsilon Value')
    plt.plot(epsilons, fl_vi_reward, label='Value Iteration')
    plt.plot(epsilons, fl_pi_reward, label='Policy Iteration')
    plt.xlabel('Epsilon')
    plt.legend()
    plt.savefig('charts/frozen_lake_reward.png')
    plt.show()
    plt.clf()

    plt.title('Taxi Reward and Epsilon Value')
    plt.plot(epsilons, tax_vi_reward, label='Value Iteration')
    plt.plot(epsilons, tax_pi_reward, label='Policy Iteration')
    plt.xlabel('Epsilon')
    plt.legend()
    plt.savefig('charts/taxi_reward.png')
    plt.show()
    plt.clf()

    plt.title('Frozen Lake Convergence Step')
    plt.xlabel('Epsilon')
    plt.ylabel('Step')
    plt.plot(fl_vi_c_step, label='Value Iteration')
    plt.plot(fl_pi_c_step, label='Policy Iteration')
    plt.legend()
    plt.savefig('charts/frozen_lake_con_step.png')
    plt.show()
    plt.clf()

    plt.title('Frozen Lake Convergence Time')
    plt.xlabel('Epsilon')
    plt.ylabel('Time(s)')
    plt.plot(fl_vi_c_time, label='Value Iteration')
    plt.plot(fl_pi_c_time, label='Policy Iteration')
    plt.legend()
    plt.savefig('charts/frozen_lake_con_time.png')
    plt.show()
    plt.clf()

    plt.title('Taxi Convergence Step')
    plt.xlabel('Epsilon')
    plt.ylabel('Step')
    plt.plot(tax_vi_c_step, label='Value Iteration')
    plt.plot(tax_pi_c_step, label='Policy Iteration')
    plt.legend()
    plt.savefig('charts/taxi_con_step.png')
    plt.show()
    plt.clf()

    plt.title('Taxi Convergence Time')
    plt.xlabel('Epsilon')
    plt.ylabel('Time(s)')
    plt.plot(tax_vi_c_time, label='Value Iteration')
    plt.plot(tax_pi_c_time, label='Policy Iteration')
    plt.legend()
    plt.savefig('charts/taxi_con_time.png')
    plt.show()
    plt.clf()
