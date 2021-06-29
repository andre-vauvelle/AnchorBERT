import pandas as pd
import matplotlib.pyplot as plt
import math

# import seaborn as sns
# code = '411.2'
name = 'bert_250.2.tsv'
# name = 'all_250.2.tsv'
# df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/{}'.format(name), sep='\t')
# df_a = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/all_anchor_714.0|714.1.tsv', sep='\t')
df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/bert_250.2.tsv.tsv', sep='\t')


# df.loc[:, 'anchor'] = df_a.anchor

print(df.shape)

# df = df.loc[:, ['IID', 'anchor', 'binomial_r']]
df_melt = df.melt(id_vars=['IID'])
grouped = df_melt.groupby('variable')
rowlength = math.ceil(grouped.ngroups)  # fix up if odd number of groups

fig, axs = plt.subplots(figsize=(9, 4),
                        nrows=1, ncols=rowlength,  # fix as above
                        gridspec_kw=dict(hspace=0.4),
                        sharey=True)  # Much control of gridspec
try:
    axs = axs.flatten()
except:
    axs = [axs]

targets = zip(grouped.groups.keys(), axs)
for i, (key, ax) in enumerate(targets):
    ax.hist(grouped.get_group(key).value, bins=200)
    ax.set_ylim([1, 200_000])
    # ax.set_xlim([0, 1])
    ax.set_title(key)
    ax.set_yscale('log')
    # ax.tick_params(axis='y', labelleft=False)
plt.setp(axs[0], ylabel='Number of Patients')
plt.setp(axs[0], xlabel='Phenotype probability')
plt.gcf().set_dpi(300)
fig.suptitle('Histogram of Phenotyping predictions 0-1 (each bar is 0.005) for {}'.format(name))
plt.show()

df_melt.loc[:, 'rounded'] = df_melt.value.round()
df_melt.groupby('variable').rounded.sum()
df_melt.groupby('variable').rounded.count()
df_melt.groupby('variable').rounded.sum() / df_melt.groupby('variable').rounded.count()

df
