import os

import matplotlib.pyplot as plt
import pandas as pd

from definitions import RESULTS_DIR

plt.rcParams['figure.figsize'] = [8.0, 8.0]
plt.rcParams['figure.dpi'] = 200

# name = 'beta3'
name = 'default'
# name = 'bert'
df = pd.read_csv(os.path.join(RESULTS_DIR, 'simulation_results/{}.csv'.format(name)))

plot_settings = {'sharey': True, 'grid': True, 'layout': (1, 7), 'figsize': (10, 10)}
# plot_settings = {'grid': True, 'figsize': (10, 10)}

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
df.groupby('name').boxplot(column='power', **plot_settings)
plt.suptitle('Power for {}'.format(name))
plt.show()
df.groupby('name').boxplot(column='auroc', **plot_settings)
plt.suptitle('AUROC for {}'.format(name))
plt.show()
df.groupby('name').boxplot(column='average_y', **plot_settings)
plt.suptitle('Average y counts for {}'.format(name))
plt.show()
