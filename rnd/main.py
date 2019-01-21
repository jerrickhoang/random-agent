import matplotlib
matplotlib.use('Agg')  # need to have this at the beginning of the file.

import click
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


def generate_drift_diffusion_plot(values, upperbound, lowerbound, tick_max=True, **kwargs):
    #plot timeseries
    ax = plt.gca()
    plt.plot(values, **kwargs)
    
    #format plot
    yticks = [lowerbound, upperbound]
    yticks_labels = [str(lowerbound), str(upperbound)]
    if tick_max:
        yticks.append(np.max(values))
        yticks_labels.append(str(np.max(values)))
    ax.set_ylim(lowerbound, upperbound)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    ax.set_xlim(0, len(values))
    ax.set_xlabel('time')
    ax.spines['right'].set_visible(False)
    
    return ax

def compute_cumulative_rewards(rewards):
    cumulative_rewards = [rewards[0]]
    for i in range(1, len(rewards)):
        total_reward = cumulative_rewards[i-1] + rewards[i]
        cumulative_rewards.append(total_reward)
    return cumulative_rewards

def generate_cum_sum_plot(exp_dir, results, title='cumulative_reward'):
    cum_sums = np.array([compute_cumulative_rewards(rewards) for rewards in results])
    cum_sums_df = pd.DataFrame(cum_sums.T)
    
    max_reward = np.max(cum_sums)
    #plot individual timeseries
    cum_sums_df.apply(generate_drift_diffusion_plot, upperbound=max_reward, lowerbound=0, color='black', alpha=0.2);
    
    #plot mean timeseries
    ax = generate_drift_diffusion_plot(np.mean(cum_sums_df, axis=1), upperbound=max_reward, lowerbound=0, 
                                       color='black', lw=3);
    fig = ax.get_figure()
    output_file_name = '{}.png'.format(title)
    fig.savefig(os.path.join(exp_dir, output_file_name))


def generate_histogram(exp_dir, data, title='hist'):
    data_pd = pd.DataFrame(data=data, columns=['data'])
    ax = data_pd.hist(bins=50)
    assert len(ax) == 1
    assert len(ax[0]) == 1
    fig = ax[0][0].get_figure()
    output_file_name = '{}.png'.format(title)
    fig.savefig(os.path.join(exp_dir, output_file_name))

def generate_plots(exp_dir, results, it_terms):

    # generate histogram for total rewards.
    total_rewards = [sum(rewards) for rewards in results]
    generate_histogram(exp_dir, total_rewards, title='total_reward_hist')
    plt.cla()

    # generate histogram for number of iterations took before termination
    generate_histogram(exp_dir, it_terms, title='it_term_hist')
    plt.cla()

    # generate cumulative sum of rewards over time
    generate_cum_sum_plot(exp_dir, results)
    plt.cla()


def run_experiment(env_name, num_iterations=1000):
    env = gym.make(env_name)
    env.reset()
    total_reward, it_term, done = 0, 0, False
    results = [0 for _ in range(num_iterations)]
    done = False
    for it in range(num_iterations):
        r = 0
        if not done:
            s, r, done, _ = env.step(env.action_space.sample())
            total_reward += r
        else:
            if it_term == 0:
                it_term = it
        results[it] = r
    return results, it_term


@click.command()
@click.option('--env_name', default='CartPole-v0', help='The name of the environemnet')
@click.option('--num_tries', default=100, help='Number of tries for each environment')
@click.option('--num_iters', default=100, help='The number of iterations to run per experiment')
def main(env_name, num_tries, num_iters):
    # Create the directory to store experiment results
    exp_name = '{}_{}'.format(env_name, str(num_tries))
    exp_dir = 'experiments/{}'.format(exp_name)
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # run num_tries number of experiments
    results = []
    it_terms = []
    for i in range(num_tries):
        rewards, it_term = run_experiment(env_name, num_iters)
        it_terms.append(it_term)
        results.append(rewards)
    np.save(os.path.join(exp_dir, 'results.dat'), results)

    generate_plots(exp_dir, results, it_terms)


if __name__ == "__main__":
    main()
