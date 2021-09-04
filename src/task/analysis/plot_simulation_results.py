import os

import matplotlib.pyplot as plt
import pandas as pd

from definitions import RESULTS_DIR


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [8.0, 8.0]
    plt.rcParams['figure.dpi'] = 200

    # name = 'beta3'
    name = 'default'
    # name = 'bert'
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'simulation_results/{}.csv'.format(name)))

    plot_settings = {'sharey': True, 'grid': True, 'layout': (1, 7), 'figsize': (10, 10)}


    # plot_settings = {'grid': True, 'figsize': (10, 10)}

    def get_power(df_group, significance_level=0.05, p_val_col='p_val'):
        total_significant = sum(df_group[p_val_col] <= significance_level)
        total_trails = df_group.shape[0]
        return total_significant / total_trails


    df.groupby('name').apply(lambda x: get_power(x, p_val_col='p_val'))
    df.groupby('name').apply(lambda x: get_power(x, p_val_col='p_val_est_ir'))
    df.groupby('name').apply(lambda x: get_power(x, p_val_col='p_val_rounded'))

    df.groupby('name').boxplot(column='odds_ratio', **plot_settings)
    plt.suptitle('Odds ratio for {}'.format(name))
    plt.show()

    df.groupby('name').boxplot(column='prevalence', **plot_settings)
    plt.suptitle('Prevalence for {}'.format(name))
    plt.show()

    df.groupby('name').boxplot(column='p_val', **plot_settings)
    plt.suptitle('P-Value for {}'.format(name))
    plt.yscale('log')
    plt.show()

    df.groupby('name').boxplot(column='p_val_est_ir', **plot_settings)
    plt.suptitle('P-Value for {}'.format(name))
    plt.yscale('log')
    plt.show()

    df.groupby('name').boxplot(column='auroc', **plot_settings)
    plt.suptitle('AUROC for {}'.format(name))
    plt.show()

    df.groupby('name').boxplot(column='average_y', **plot_settings)
    plt.suptitle('Average y counts for {}'.format(name))
    plt.show()
