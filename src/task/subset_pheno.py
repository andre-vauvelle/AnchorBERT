import argparse
import os
import pandas as pd


# parser = argparse.ArgumentParser(description='Subset file by total patients or cases by a fraction of total n times')
#
# parser.add_argument('-p', '--phenofile_path', help='Full path to phenofile')
# parser.add_argument('-s', '--subset-frac', help='Fraction of data to retain between 0-1',
#                     type=float)
# parser.add_argument('-c', '--cases', default=False, help='Subsample only cases', action='store_true')
# parser.add_argument('-t', '--trials', default=1, help='Number of trails/copies', type=int)
#
# args = parser.parse_args()
#
# subset_frac = args.subset_frac
# trials = args.trials
# cases_s = args.cases
# phenofile_path = args.phenofile_path
#

def subset_pheno(subset_frac, trials, cases_s, phenofile_path):
    """
    Splits phenotype file up into sub-sampled files
    :param subset_frac:
    :param args:
    :return:
    """
    print(subset_frac, 'c' if cases_s else '')
    phenofile = pd.read_csv(phenofile_path, sep='\t')

    for trial in range(trials):
        if cases_s:
            phenofile_cases_col = phenofile.loc[:, 'threshold1']
            cases = phenofile[phenofile_cases_col == 1]
            controls = phenofile[phenofile_cases_col != 1]
            sampled_cases = cases.sample(frac=subset_frac, axis=0)
            new_phenofile = pd.concat([sampled_cases, controls], axis=0)
            new_phenofile = new_phenofile.sample(frac=1)  # Shuffle
        else:
            new_phenofile = phenofile.sample(frac=subset_frac, axis=0)

        filename = phenofile_path[:-4] + '_{}{}/{}'.format(
            'c' if cases_s else '',
            str(round(subset_frac, 1)),
            str(trial))
        os.makedirs(filename, exist_ok=True)
        new_phenofile.to_csv(filename + '.tsv',
                             sep='\t', index=False)

# if __name__ == '__main__':
#     subset_pheno(subset_frac, trials, cases_s, phenofile_path)
