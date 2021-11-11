import itertools
import os
import subprocess

import matplotlib.ticker as mtick
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from definitions import DATA_DIR, RESULTS_DIR
from task.analysis.snp_assoc_catalog import merge_on_ld_and_id

from task.subset_pheno import subset_pheno

plt.rcParams['figure.figsize'] = [9.0, 8.0]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['font.size'] = 22


def get_data_ablation_value(file, control_flag=False):
    """
    Extracts 0.2 from something like chr9_411.2_anchor_0.2.tsv.logreg_anchor.glm.linear

    :param file: filename of gwas results
    :return: data ablation value
    """
    if control_flag:
        return file.split('anchor_c')[1][:3]
    else:
        return file.split('anchor')[1][1:4]


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


def is_c_flag(file, filter_term):
    return 'c' == file.split(filter_term)[1][1]


def plink_trails_results(filter_term='411.2_ca0_co0_anchor',
                         results_dir='/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/',
                         control_flag=False):
    files = os.listdir(results_dir)
    files = [file for file in files if file.split('.')[-1] != 'log' and 'chr' in file]  # remove log files
    files = [file for file in files if filter_term in file]
    if control_flag:
        files = [file for file in files if is_c_flag(file, filter_term)]
    else:
        files = [file for file in files if not is_c_flag(file, filter_term)]
    # phenos = set([get_pheno(file) for file in files])

    store = []
    for i, file in tqdm(enumerate(files), desc='Loading Files'):
        try:
            df = pd.read_csv(os.path.join(results_dir, file), sep='\t', error_bad_lines=False)
            df.loc[:, 'trial'] = get_trial(file)
            df.loc[:, 'phenotype'] = get_pheno(file)
            df.loc[:, 'ablation'] = get_data_ablation_value(file, control_flag=control_flag)
            store.append(df)
        except pd.errors.EmptyDataError:
            print('\n', file)

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
        '428.2': {
            'name': 'Heart Failure',
            'filter_term': '428.2_ca0_co0_anchor',
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/428_catalog.tsv",

        },
        '411.2': {
            'name': 'Myocardial Infarction',
            'filter_term': '411.2_ca0_co0_anchor',
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/411_catalog.tsv",

        },
        '250.2': {
            'name': 'Type 2 Diabetes',
            'filter_term': '250.2_ca0_co0_anchor',
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/250_catalog.tsv",
        },
        '714.0|714.1': {
            'name': 'Rheumatoid Arthritis',
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/714_catalog.tsv",
            'filter_term': '714.0|714.1_ca0_co0_anchor',
        },
        '290.1': {
            'name': 'Dementia',
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/290_catalog.tsv",
            'filter_term': '290.1_ca0_co0_anchor',
        }
    }

    method_dict = {
        "bert_anchor": 'AnchorBERT',
        'logreg_anchor': 'Anchor LR',
        'binomial_r': 'Pheprob',
        'threshold1': "Threshold 1",
        'threshold2': "Threshold 2",
        'threshold3': "Threshold 3"
    }

    ld_path = '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/ld/ld0.5_collection.tab.gz'
    # code = '428.2'

    ld_panel = pd.read_csv(ld_path, sep='\t')
    ld_panel.loc[:, 'chrom'] = ld_panel.snpID.str.split(':').str[0].astype(int)
    ld_panel.loc[:, 'pos'] = ld_panel.snpID.str.split(':').str[1].astype(int)

    p_cut = 5e-8
    code = '411.2'
    control_flag = True
    # code = '428.2'
    # code = '290.1'
    # code = '714.0|714.1'
    # codes = ['411.2', '250.2', '290.1', '714.0|714.1', '428.2']
    codes = ['250.2', ]

    control = [True, False]
    code_control = itertools.product(codes, control)

    for code, control_flag in code_control:
        c = 'c' if control_flag else ''
        filter_term = pheno_dict[code]['filter_term']
        catalog_path = pheno_dict[code]['catalog_path']
        control_string = 'control' if control_flag else 'both'

        catalog = pd.read_csv(catalog_path, sep='\t')
        catalog.columns = catalog.columns.str.lower().str.replace(' ', '_')
        catalog.chr_id = pd.to_numeric(catalog.chr_id, errors='coerce', downcast='integer')
        catalog.chr_pos = pd.to_numeric(catalog.chr_pos, errors='coerce', downcast='integer')
        catalog.dropna(subset=['chr_id', 'chr_pos'], inplace=True)
        catalog.rename(columns={'chr_id': 'chrom', 'chr_pos': 'pos'}, inplace=True)
        catalog = catalog[catalog['p-value'] <= 5e-8]
        catalog_ld_pos = catalog.merge(
            ld_panel.loc[:, ['chrom', 'pos', 'loci_upstream', 'loci_downstream']],
            on=['chrom', 'pos'])
        catalog_ld_id = catalog.merge(ld_panel.loc[:, ['rsID', 'loci_upstream', 'loci_downstream']],
                                      left_on='snps', right_on='rsID', how='left')
        catalog_ld = pd.concat([catalog_ld_pos, catalog_ld_id], axis=0)
        catalog_ld = catalog_ld.loc[:, ['rsID', 'snps', 'chrom', 'pos', 'loci_upstream', 'loci_downstream']]

        df_all = plink_trails_results(filter_term=filter_term, control_flag=control_flag)
        df_all.ablation = df_all.ablation.astype(float)
        df_all.rename(columns={'#CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        df_all = df_all.loc[:, ['chrom', 'pos', 'ID', 'P', 'trial', 'phenotype', 'ablation']]

        df_all.to_csv(os.path.join(RESULTS_DIR, 'ablation_results', '{}_{}.tsv'.format(code, control_string)))
        # df_all = pd.read_csv(
        #     os.path.join(RESULTS_DIR, 'ablation_results', '{}_{}.tsv'.format(code, control_string)),
        # )

        check_counts = df_all.groupby(['phenotype', 'ablation', 'trial']).P.count()
        if not (check_counts == check_counts.iloc[0]).all():
            print('Not all completed')

        df_all_sig = df_all[df_all.P <= p_cut]

        catalog_snps = catalog.snps.str.split(' x ').explode().unique()
        df_all_ld = df_all_sig.merge(ld_panel.loc[:, ['chrom', 'pos', 'loci_upstream', 'loci_downstream']],
                                     on=['chrom', 'pos'], how='left')

        merge_store = []
        for (phenotype_name, ablation, trial), df in df_all_ld.groupby(['phenotype', 'ablation', 'trial']):
            # print(df.shape)
            df_m = merge_on_ld_and_id(df, catalog_ld)
            merge_store.append(df_m)

        df_results = pd.concat(merge_store, axis=0)

        ############
        # # def plot_ablation
        #############

        phenotype_name = pheno_dict[code]['name']
        threshold1_snps = df_results[(df_results.phenotype == 'threshold1') &
                                     (df_results.ablation == 1.0)].rsID.unique()
        df_results_sub = df_results  # [df_results.rsID.isin(threshold1_snps) & df_results.trial.isin(range(0,8))]

        total_snps = df_results_sub.snps.unique().shape[0]

        counts_p_trial = df_results_sub.groupby(['phenotype', 'ablation', 'trial']).apply(
            lambda x: (x.snps.unique().shape[0]) / total_snps).rename('counts').reset_index()
        counts_p_mean = counts_p_trial.groupby(['phenotype', 'ablation'])['counts'].mean()
        counts_p_std = counts_p_trial.groupby(['phenotype', 'ablation'])['counts'].std()
        counts_p_min = counts_p_trial.groupby(['phenotype', 'ablation'])['counts'].min()
        counts_p_max = counts_p_trial.groupby(['phenotype', 'ablation'])['counts'].max()

        df_results_sub_grouped = counts_p_mean.reset_index()
        df_results_sub_grouped.loc[:, 'counts_std'] = counts_p_std.values
        df_results_sub_grouped.loc[:, 'counts_min'] = counts_p_min.values
        df_results_sub_grouped.loc[:, 'counts_max'] = counts_p_max.values
        df_results_sub_grouped.fillna(0, inplace=True)
        # Add zeros for results not returned
        df_results_sub_grouped_blank = pd.DataFrame(
            itertools.product(method_dict.keys(), np.arange(0, 1.1, 0.1)),
            columns=['phenotype', 'ablation']
        )
        df_results_sub_grouped = pd.concat([df_results_sub_grouped, df_results_sub_grouped_blank], axis=0).fillna(
            0).sort_values('counts')
        df_results_sub_grouped.ablation = df_results_sub_grouped.ablation.round(1)
        df_results_sub_grouped = df_results_sub_grouped.drop_duplicates(
            subset=['ablation', 'phenotype'], keep='last')
        marker = itertools.cycle(('o', 'X', 'P', '*', 'D', 'v'))
        linestyle = itertools.cycle(('-', '--', '-.', ':', ':', ':'))
        df_results_sub_grouped.to_csv(
            os.path.join(RESULTS_DIR, 'ablation_results_plots', 'data_{}{}_anchor.csv'.format(code, c)))
        for pheno, df in df_results_sub_grouped.groupby('phenotype'):
            df.sort_values('ablation', inplace=True, ascending=False)
            plt.plot(100 * df.ablation, 100 * df.counts, label=method_dict[pheno], marker=next(marker),
                     linestyle=next(linestyle))
            plt.fill_between(100 * df.ablation,
                             100 * df.counts - 100 * df.counts_std,
                             100 * df.counts + 100 * df.counts_std,
                             alpha=0.3)
            # plt.fill_between(100 * df.ablation,
            #                  100 * df.counts_min,
            #                  100 * df.counts_max,
            #                  alpha=0.3)
            # plt.errorbar(-100*df.ablation, 100*df.counts, yerr=100*df.counts_std,
            #              capsize=5,
            #              label=method_dict[pheno],
            #              dash_capstyle='butt')

        x_axis_label = '% of total patients' if not control_flag else '% of total cases'
        plt.ylabel('% of all significant SNPs \n from all methods')
        plt.xlabel(x_axis_label)
        plt.title('{}'.format(phenotype_name))
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(np.arange(0, 110, 10))
        ax = plt.gca()
        ax.invert_xaxis()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 1, 3, 4, 5]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], frameon=False,
                   prop={'size': 18})
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()

        plt.savefig(os.path.join(RESULTS_DIR, 'ablation_results_plots', '{}{}_anchor.png'.format(code, c)))
        plt.clf()
