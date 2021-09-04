import itertools
import os
import subprocess

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from definitions import DATA_DIR, RESULTS_DIR


# from task.subset_pheno import subset_pheno


def get_data_ablation_value(file):
    """
    Extracts 0.2 from something like chr9_411.2_anchor_0.2.tsv.logreg_anchor.glm.linear

    :param file: filename of gwas results
    :return: data ablation value
    """
    return file.split('anchor_')[1][:3]


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


def run_plink(pheno_dir, chr_snps_dict, trials, command='qsub', controls_only=False):
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
    c_flag = 'c' if controls_only else ''

    for chr, snps in chr_snps_dict.items():
        pfile = str(chr)
        snps_str = ','.join(snps)
        subprocess.run([command, PLINK_SH_PATH, pfile, pheno_dir, snps_str, str(trials), c_flag])


def plink_trails_results(filter_term='(?=.*foo)(?=.*baz)',
                         results_dir='/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/'):
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
        df.loc[:, 'ablation'] = get_data_ablation_value(file)
        store.append(df)

    df_total = pd.concat(store, axis=0)

    df_total.P = pd.to_numeric(df_total.P, errors='coerce')
    df_total.loc[:, 'log_pval'] = -np.log10(df_total.P.values)

    return df_total


def in_range(row, catalog):
    """
    checks if snp is in catalog using ld0.5 range upstream and downstream
    :param row:
    :param catalog:
    :return: catalog rows which match
    """
    chrom = catalog[catalog.chrom == row['chrom']]
    return ((chrom.pos <= row.loci_downstream) & (chrom.pos >= row.loci_upstream)).any(axis=0)


def in_range_snps(row, catalog):
    """
    checks if snp is in catalog using ld0.5 range upstream and downstream
    :param row:
    :param catalog:
    :return: catalog rows which match
    """
    chrom = catalog[catalog.chrom == row['chrom']]
    return chrom[(chrom.pos <= row.loci_downstream) & (chrom.pos >= row.loci_upstream)].snps.unique()


def get_matched_data_results(sig_gwas_ld, catalog_ld):
    joined = merge_on_ld_and_id(sig_gwas_ld, catalog_ld)
    matched_gwas_ids = set(joined.ID.dropna().unique())
    matched_cata_ids = set(joined.rsID.dropn().unique())

    gwas_ids = set(sig_gwas_ld.ID.unique())
    cata_ids = set(catalog_ld.rsID.unique())

    unmatched_gwas_ids = gwas_ids - matched_gwas_ids
    unmatched_cata_ids = cata_ids - matched_cata_ids

    data = {
        "matched_gwas_ids": matched_gwas_ids,
        "matched_cata_ids": matched_cata_ids,
        "gwas_ids": gwas_ids,
        "cata_ids": cata_ids,
        "unmatched_gwas_ids": unmatched_gwas_ids,
        "unmatched_cata_ids": unmatched_cata_ids
    }
    return data


def merge_on_ld_and_id(sig_gwas_ld, catalog_ld):
    """Both catalog_ld and sig_gwas_ld must have been merged with ld"""

    s_chrom = sig_gwas_ld.chrom.values
    c_chrom = catalog_ld.chrom.values
    s_pos = sig_gwas_ld.pos.values
    c_pos = catalog_ld.pos.values
    s_upper = sig_gwas_ld.loci_upstream.values
    s_down = sig_gwas_ld.loci_downstream.values
    c_upper = catalog_ld.loci_upstream.values
    c_down = catalog_ld.loci_downstream.values

    s_id = sig_gwas_ld.ID.values
    c_id = catalog_ld.snps.values

    # Create join matrix, rows are sig_gwas_ld, columns catalog
    i, j = np.where(
        (
                s_pos[:, None] >= c_upper) & (s_pos[:, None] <= c_down) & (s_chrom[:, None] == c_chrom)
        | (
                s_upper[:, None] >= c_pos) & (s_down[:, None] <= c_pos) & (s_chrom[:, None] == c_chrom) | (
                s_id[:, None] == c_id
        )

    )

    joined = pd.DataFrame(
        np.column_stack([sig_gwas_ld.values[i], catalog_ld.values[j]]),
        columns=sig_gwas_ld.columns.append(catalog_ld.columns)
    )
    return joined


if __name__ == '__main__':

    ld_path = '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/ld/ld0.5_collection.tab.gz'
    # https: // data.broadinstitute.org / mpg / snpsnap / database_download.html

    # build_path = '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/ld/hg19_avsnp147.txt.gz'
    # http: // www.openbioinformatics.org / annovar / download /

    pheno_dict = {
        '250.2': {
            'name': 'Type 2 Diabetes',
            'big_gwas_thr1': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.threshold1.gz",
            'big_gwas_thr2': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.threshold2.gz",
            'big_gwas_thr3': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.threshold3.gz",
            'big_gwas_logreg': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.logreg_anchor.gz",
            'big_gwas_bert': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.bert_anchor.gz",
            'big_gwas_binomial': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/250.2_ca0_co0_anchor.tsv.binomial_r.gz",
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/250_catalog.tsv",
            #     https://www.ebi.ac.uk/gwas/efotraits/EFO_0001360
        },
        '411.2': {
            'name': 'Myocardial Infarction',
            'big_gwas_thr1': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.threshold1.gz",
            'big_gwas_thr2': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.threshold2.gz",
            'big_gwas_thr3': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.threshold3.gz",
            'big_gwas_logreg': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.logreg_anchor.gz",
            'big_gwas_bert': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.bert_anchor.gz",
            'big_gwas_binomial': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/411.2_ca0_co0_anchor.tsv.binomial_r.gz",
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/411_catalog.tsv",
            #     https://www.ebi.ac.uk/gwas/efotraits/EFO_0000612
        },
        '428.2': {
            'name': 'Heart Failure',
            'big_gwas_thr1': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.threshold1.gz",
            'big_gwas_thr2': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.threshold2.gz",
            'big_gwas_thr3': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.threshold3.gz",
            'big_gwas_logreg': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.logreg_anchor.gz",
            'big_gwas_bert': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.bert_anchor.gz",
            'big_gwas_binomial': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/428.2_ca0_co0_anchor.tsv.binomial_r.gz",
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/428_catalog.tsv",
            #     https://www.ebi.ac.uk/gwas/efotraits/EFO_0003144
        },
        '714.0|714.1': {
            'name': 'Rheumatoid Arthritis',
            'big_gwas_thr1': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.threshold1.gz",
            'big_gwas_thr2': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.threshold2.gz",
            'big_gwas_thr3': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.threshold3.gz",
            'big_gwas_logreg': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.logreg_anchor.gz",
            'big_gwas_bert': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.bert_anchor.gz",
            'big_gwas_binomial': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/714.0|714.1_ca0_co0_anchor.tsv.binomial_r.gz",
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/714_catalog.tsv",
            # https://www.ebi.ac.uk/gwas/efotraits/EFO_0000685
        },
        '290.1': {
            'name': 'Dementia',
            'big_gwas_thr1': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.threshold1.gz",
            'big_gwas_thr2': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.threshold2.gz",
            'big_gwas_thr3': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.threshold3.gz",
            'big_gwas_logreg': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.logreg_anchor.gz",
            'big_gwas_bert': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.bert_anchor.gz",
            'big_gwas_binomial': "/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/combined/290.1_ca0_co0_anchor.tsv.binomial_r.gz",
            'catalog_path': "/SAN/ihibiobank/denaxaslab/andre/UKBB/data/raw/gwas_catalog/290_catalog.tsv",
            # https://www.ebi.ac.uk/gwas/efotraits/HP_0000726
        }
    }

    codes = ['411.2', '250.2', '714.0|714.1', '428.2', '290.1']
    # code = '411.2'
    # code = '250.2'
    # code = '714.0|714.1'
    # code = '428.2'
    code = '290.1'

    for code in codes:
        big_gwas_thr1_path = pheno_dict[code]['big_gwas_thr1']
        big_gwas_thr2_path = pheno_dict[code]['big_gwas_thr2']
        big_gwas_thr3_path = pheno_dict[code]['big_gwas_thr3']
        big_gwas_logreg_path = pheno_dict[code]['big_gwas_logreg']
        big_gwas_bert_path = pheno_dict[code]['big_gwas_bert']
        big_gwas_binomial_path = pheno_dict[code]['big_gwas_binomial']
        catalog_path = pheno_dict[code]['catalog_path']

        big_gwas_thr1 = pd.read_csv(big_gwas_thr1_path, sep='\t', compression='gzip', error_bad_lines=False)
        big_gwas_thr2 = pd.read_csv(big_gwas_thr2_path, sep='\t', compression='gzip', error_bad_lines=False)
        big_gwas_thr3 = pd.read_csv(big_gwas_thr3_path, sep='\t', compression='gzip', error_bad_lines=False)
        big_gwas_logreg = pd.read_csv(big_gwas_logreg_path, sep='\t', compression='gzip', error_bad_lines=False)
        big_gwas_bert = pd.read_csv(big_gwas_bert_path, sep='\t', compression='gzip', error_bad_lines=False)
        big_gwas_binomial = pd.read_csv(big_gwas_binomial_path, sep='\t', compression='gzip', error_bad_lines=False)

        ld_panel = pd.read_csv(ld_path, sep='\t')
        ld_panel.loc[:, 'chrom'] = ld_panel.snpID.str.split(':').str[0].astype(int)
        ld_panel.loc[:, 'pos'] = ld_panel.snpID.str.split(':').str[1].astype(int)

        catalog = pd.read_csv(catalog_path, sep='\t')
        catalog.columns = catalog.columns.str.lower().str.replace(' ', '_')
        catalog.chr_id = pd.to_numeric(catalog.chr_id, errors='coerce', downcast='integer')
        catalog.chr_pos = pd.to_numeric(catalog.chr_pos, errors='coerce', downcast='integer')
        catalog.dropna(subset=['chr_id', 'chr_pos'], inplace=True)

        # rename all chrm and pos columns to ld panel standard
        catalog.rename(columns={'chr_id': 'chrom', 'chr_pos': 'pos'}, inplace=True)
        big_gwas_thr1.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        big_gwas_thr2.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        big_gwas_thr3.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        big_gwas_logreg.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        big_gwas_bert.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)
        big_gwas_binomial.rename(columns={'CHROM': 'chrom', 'POS': 'pos'}, inplace=True)

        catalog_significant = catalog[catalog['p-value'] <= 5e-8]
        catalog_ld_pos = catalog_significant.merge(
            ld_panel.loc[:, ['chrom', 'pos', 'loci_upstream', 'loci_downstream']],
            on=['chrom', 'pos'])
        catalog_ld_id = catalog_significant.merge(ld_panel.loc[:, ['rsID', 'loci_upstream', 'loci_downstream']],
                                                  left_on='snps', right_on='rsID', how='left')
        catalog_ld = pd.concat([catalog_ld_pos, catalog_ld_id], axis=0)

        gwas_names = ['bert_anchor', 'logreg_anchor', 'binomial', 'threshold1', 'threshold2', 'threshold3']
        gwas_list = [big_gwas_bert, big_gwas_logreg, big_gwas_binomial, big_gwas_thr1, big_gwas_thr2, big_gwas_thr3]

        store = []
        for gwas, name in zip(gwas_list, gwas_names):
            sig_gwas = gwas[gwas.P <= 5e-8]
            sig_gwas_ld = sig_gwas.merge(ld_panel.loc[:, ['chrom', 'pos', 'loci_upstream', 'loci_downstream']],
                                         on=['chrom', 'pos'], how='left')
            if sig_gwas_ld.shape[0] == 0:
                store.append({
                    'name': name,
                })
            else:
                data = merge_on_ld_and_id(sig_gwas_ld, catalog_ld)
                store.append({
                    'name': name,
                    "len_gwas_ids": len(data['gwas_ids']),
                    "len_cata_ids": len(data['cata_ids']),
                    "len_matched_gwas_ids": len(data['matched_gwas_ids']),
                    "len_matched_cata_ids": len(data['matched_cata_ids']),
                    "len_unmatched_gwas_ids": len(data['unmatched_gwas_ids']),
                    "len_unmatched_cata_ids": len(data['unmatched_cata_ids']),
                    "gwas_ids": data['gwas_ids'],
                    "cata_ids": data['cata_ids'],
                    "matched_gwas_ids": data['matched_gwas_ids'],
                    "matched_cata_ids": data['matched_cata_ids'],
                    "unmatched_gwas_ids": data['unmatched_gwas_ids'],
                    "unmatched_cata_ids": data['unmatched_cata_ids'],
                    "mean_cata_pvalues_in_gwas_top5": sig_gwas[sig_gwas.ID.isin(data['matched_gwas_ids'])][
                        'P'].sort_values().head(5).mean(),
                    "mean_cata_pvalues_in_gwas": sig_gwas[sig_gwas.ID.isin(data['matched_gwas_ids'])][
                        'P'].sort_values().mean()
                })

        len_cols = ['name',
                    "len_gwas_ids",
                    "len_cata_ids",
                    "len_matched_gwas_ids",
                    "len_matched_cata_ids",
                    "len_unmatched_gwas_ids",
                    "len_unmatched_cata_ids",
                    "mean_cata_pvalues_in_gwas_top5",
                    "mean_cata_pvalues_in_gwas"
                    ]

        df_results = pd.DataFrame(store)
        df_results.loc[:, len_cols]
        df_results.to_csv(os.path.join(RESULTS_DIR, 'catalog', '{}.csv'.format(code)), index=False)

    codes = ['411.2', '250.2', '714.0|714.1', '428.2', '290.1']
    code = '428.2'
    code = '411.2'
    code = '290.1'
    store = []
    for code in codes:
        df_results = pd.read_csv(os.path.join(RESULTS_DIR, 'catalog', '{}.csv'.format(code)))
        df_results.loc[:, 'code'] = code
        store.append(df_results.loc[:, len_cols + ['code']])

    df_results_all = pd.concat(store, axis=0)
    df_results_all.loc[:, 'power'] = df_results_all.len_matched_cata_ids / df_results_all.len_cata_ids

    df_results_all.loc[:, ['code', 'name', 'power', 'len_gwas_ids', 'len_cata_ids', 'len_matched_cata_ids']]
