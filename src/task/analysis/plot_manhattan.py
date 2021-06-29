import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from qmplot import manhattanplot

from definitions import RESULTS_DIR


def missing_chr(files):
    chrs = set([int(file.split('_')[0].strip('chr')) for file in files])
    chrs_missing = set(range(1, 21 + 1)) - chrs
    print("Missing Chromosomes {}".format(chrs_missing))


def rename_files(files, new_name):
    results_dir = '/SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/'
    for i,file in enumerate(files):
        os.rename(os.path.join(results_dir, file), os.path.join(results_dir, new_name + '_{}'.format(i)))


df = pd.read_table(os.path.join(RESULTS_DIR, 'gwas_results', 'combined', 'bert_250.gz'), compression='gzip')


if __name__ == "__main__":
    save_term = '714'
    filter_term = '.*?714.*'
    results_dir = '/SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/'
    files = os.listdir(results_dir)
    files = [file for file in files if file.split('.')[-1] != 'log' and 'chr' in file]  # remove log files
    files = [file for file in files if re.match(filter_term, file)]
    phenos = list(set([file.split('.')[-3] for file in files]))
    for pheno in phenos:
        pheno_files = [file for file in files if pheno in file]

        missing_chr(pheno_files)
        store = []
        for file in pheno_files:
            try:
                df = pd.read_table(os.path.join(results_dir, file), sep="\t")
                df = df.dropna(how="any", axis=0)  # clean data
                store.append(df)
            except pd.errors.EmptyDataError:
                print('EmptyDataError for file: {}'.format(file))
                pass
        pheno_df = pd.concat(store, axis=0)
        pheno_df = pheno_df.sort_values(by=['#CHROM', 'POS'], ascending=True)
        pheno_df.reset_index(drop=True, inplace=True)

        filepath = os.path.join(results_dir, '{}_{}.txt'.format(pheno, save_term))
        pheno_df.to_csv(filepath, index=False, sep='\t')
        print('Saved data at: {}'.format(filepath))
        figname = "/home/vauvelle/pycharm-sftp/pheprob/analysis/plots/{}_{}.png".format(pheno, save_term)
        ax = manhattanplot(data=pheno_df[pheno_df.P != 0], figname=figname)
        print('Saved fig at: {}'.format(figname))
