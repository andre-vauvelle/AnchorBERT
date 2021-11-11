import itertools
import os
import subprocess

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from definitions import DATA_DIR

from task.subset_pheno import subset_pheno


def get_data_ablation_value(file, control_flag=False):
    """
    Extracts 0.2 from something like chr9_411.2_anchor_0.2.tsv.logreg_anchor.glm.linear

    :param file: filename of gwas results
    :return: data ablation value
    """
    if control_flag:
        return file.split('co0_c')[1][1:4]
    else:
        return file.split('co0')[1][1:4]


def get_pheno(file):
    """
    Extracts phenotype method name from file string
    :param file: filename of gwas result
    :return: phenotype method name
    """
    return file.split('.')[-3]


def get_snp(file):
    return file.split('.')[-4].split('_')[-2]


def get_trial(file):
    return int(file.split('.')[-4].split('_')[-1])


def get_config(pheno_dir, chr_snps_dict, trials, command='qsub'):
    """
    bash src/jobs/pheno_plink_snp_trials.sh 6 714.0\|714.1_anchor rs2395185 9

    run pheno_plink_snp_trials, returns results to

    OUTPUT_FILE=chr"$1"_"$2""$frac"_"$3"_"$trial"
    PFILE=chr"$1"
    PHENO_DIR="$2"
    SNPS="$3"
    TRIALS="$4"

    --out /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/"$OUTPUT_FILE" \
    :param controls_only:
    :param command:
    :param pheno_dir:
    :param chr_snps_dict:
    :param trials:
    :return:
    """
    PLINK_SH_PATH = '/home/vauvelle/pycharm-sftp/pheprob/src/jobs/pheno_plink_snp_trials.sh'

    store = []
    for chr, snps in chr_snps_dict.items():
        pfile = str(chr)
        snps_str = ','.join(snps)
        store.append({'pfile': pfile, 'pheno_dir': pheno_dir, 'snps_str': snps_str, 'trials': trials})
        # subprocess.run([command, PLINK_SH_PATH, pfile, pheno_dir, snps_str, str(trials)])
    return store


def plink_trails_results(filter_term='(?=.*foo)(?=.*baz)',
                         results_dir='/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/',
                         control_flag=False):
    files = os.listdir(results_dir)

    files = [file for file in files if file.split('.')[-1] != 'log' and 'chr' in file]  # remove log files
    files = [file for file in files if re.match(filter_term, file)]
    files.sort(key=get_data_ablation_value, reverse=True)
    # phenos = set([get_pheno(file) for file in files])

    store = []
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(results_dir, file), sep='\t', error_bad_lines=False)
        df.loc[:, 'trial'] = get_trial(file)
        df.loc[:, 'phenotype'] = get_pheno(file)
        df.loc[:, 'ablation'] = get_data_ablation_value(file, control_flag=control_flag)
        store.append(df)

    df_total = pd.concat(store, axis=0)

    df_total.P = pd.to_numeric(df_total.P, errors='coerce')
    df_total.loc[:, 'log_pval'] = -np.log10(df_total.P.values)

    return df_total


# def plink_noise_results(filter_term='(?=.*foo)(?=.*baz)',
#                         results_dir='/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/',
#                         control_flag=False):
#     files = os.listdir(results_dir)
#
#     files = [file for file in files if file.split('.')[-1] != 'log' and 'chr' in file]  # remove log files
#     files = [file for file in files if re.match(filter_term, file)]
#     files.sort(key=get_data_ablation_value, reverse=True)
#     # phenos = set([get_pheno(file) for file in files])
#
#     store = []
#     for i, file in enumerate(files):
#         df = pd.read_csv(os.path.join(results_dir, file), sep='\t', error_bad_lines=False)
#         df.loc[:, 'trial'] = get_trial(file)
#         df.loc[:, 'phenotype'] = get_pheno(file)
#         df.loc[:, 'ablation'] = get_data_ablation_value(file, control_flag=control_flag)
#         store.append(df)
#
#     df_total = pd.concat(store, axis=0)
#
#     df_total.P = pd.to_numeric(df_total.P, errors='coerce')
#     df_total.loc[:, 'log_pval'] = -np.log10(df_total.P.values)
#
#     return df_total


if __name__ == '__main__':
    pheno_dict = {
        '411.2': {
            'name': 'Myocardial Infarction',
            'big_gwas': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.threshold1.gz",
            'catalog_results': '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/catalog/411.2.csv',
            'full_pheno_path': '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/411.2_ca0_co0_anchor.tsv',
            'pheno_dir': '411.2_ca0_co0_anchor_',

        },
        '250.2': {
            'name': 'Type 2 Diabetes',
            'big_gwas': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.threshold1.gz",
            'catalog_results': '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/catalog/250.2.csv',
            'full_pheno_path': '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/250.2_ca0_co0_anchor.tsv',
            'pheno_dir': '250.2_ca0_co0_anchor',
        },
        '714.0|714.1': {
            'name': 'Rheumatoid Arthritis',
            'big_gwas': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.threshold3.gz",
            'catalog_results': '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/catalog/714.0|714.1.csv',
            'full_pheno_path': '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/714.0|714.1_ca0_co0_anchor.tsv',
            'pheno_dir': '714.0|714.1_ca0_co0_anchor_',
        },
        '428.2': {
            'name': 'Heart Failure',
            'big_gwas': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.threshold1.gz",
            'catalog_results': '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/catalog/428.2.csv',
            'full_pheno_path': '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/428.2_ca0_co0_anchor.tsv',
            'pheno_dir': '428.2_ca0_co0_',
        },
        '290.1': {
            'name': 'Dementia',
            'big_gwas': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.threshold1.gz",
            'catalog_results': '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/catalog/290.1.csv',
            'full_pheno_path': '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/290.1_ca0_co0_anchor.tsv',
            'pheno_dir': '290.1_ca0_co0_anchor_',
        }
    }

    # code = '428.2'
    use_all_matched = True
    code = '411.2'
    trials = 10
    codes = ['411.2', '428.2', '290.1', '714.0|714.1', '250.2']
    codes = ['411.2', '250.2']
    # codes = ['290.1', '714.0|714.1', '250.2']
    code = '714.0|714.1'
    store = []
    for code in codes:
        name = pheno_dict[code]['name']
        big_gwas_path = pheno_dict[code]['big_gwas']
        catalog_path = pheno_dict[code]['catalog_results']
        pheno_dir = pheno_dict[code]['pheno_dir']

        catalog_results = pd.read_csv(catalog_path)
        catalog_results = catalog_results.dropna(subset=['matched_gwas_ids'])
        catalog_results.matched_gwas_ids = catalog_results.matched_gwas_ids.apply(lambda x: eval(x))
        matched_snps = catalog_results.matched_gwas_ids.explode().drop_duplicates().to_frame()
        big_gwas = pd.read_csv(big_gwas_path, sep='\t', compression='gzip', error_bad_lines=False)
        matched_snps = matched_snps.merge(big_gwas.loc[:, ['CHROM', 'ID']], left_on='matched_gwas_ids',
                                          right_on='ID')
        if use_all_matched:
            chr_snps_dict = matched_snps.groupby('CHROM').ID.apply(lambda x: list(set(x))).to_dict()
        else:
            big_gwas.sort_values('P', ascending=True, inplace=True)
            sig_snps = big_gwas[big_gwas.P < 5e-8].drop_duplicates()
            chr_snps_dict = sig_snps.groupby('CHROM').ID.apply(lambda x: list(set(x))).to_dict()

        # catalog_results.matched_cata_ids = catalog_results.matched_cata_ids.apply(lambda x: eval(x))
        #
        # for f in np.arange(0, 1.1, 0.1):
        #     subset_pheno(subset_frac=f, trials=10, cases_s=True, phenofile_path=pheno_dict[code]['full_pheno_path'])
        #     if f != 0:
        #         subset_pheno(subset_frac=f, trials=10, cases_s=False,
        #                      phenofile_path=pheno_dict[code]['full_pheno_path'])

        for f in np.arange(0, 1.1, 0.1):
            ablated_pheno_dir_c = pheno_dir + 'c' + str(round(f, 1))
            ablated_pheno_dir = pheno_dir + str(round(f, 1))
            for chr, snps in chr_snps_dict.items():
                pfile = str(chr)
                snps_str = ' '.join(snps)
                store.append({'pfile': pfile, 'pheno_dir': ablated_pheno_dir_c, 'snps_str': snps_str, 'trials': trials})
                if f!=0:
                    store.append({'pfile': pfile, 'pheno_dir': ablated_pheno_dir, 'snps_str': snps_str, 'trials': trials})

    df = pd.DataFrame(store)
    df.to_csv('/home/vauvelle/pycharm-sftp/pheprob/src/jobs/configs/ablation_gwas_config-{}.csv'.format(df.shape[0]),
              index=False)
