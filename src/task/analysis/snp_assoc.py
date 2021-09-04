import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from definitions import DATA_DIR


def get_data_ablation_value(file):
    """
    Extracts 0.2 from something like chr9_411.2_anchor_0.2.tsv.logreg_anchor.glm.linear

    :param file: filename of gwas results
    :return: data ablation value
    """
    return file.split('_rs')[0].split('_')[-1]


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


if __name__ == '__main__':

    phenotype_name = 'Type 2 diabetes'
    # phenotype_name = 'Myocardial Infarction'
    # phenotype_name = 'Heart Failure'
    # phenotype_name = 'Rheumatoid Arthritis'
    # (?=.*foo)(?=.*baz)
    filter_term = '(?=.*chr10_250.2_anchor_0)(?=.*rs7903146)'
    # filter_term = '(?=.*chr9_411.2_anchor_c)(?=.*rs10757277)'
    # filter_term = '(?=.*chr6_428.2_anchor_c)(?=.*rs118039278)'
    # filter_term = '(?=.*chr6_714.0\|714.1_anchor_c)(?=.*rs2395185)'
    results_dir = '/SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/'
    files = os.listdir(results_dir)

    files = [file for file in files if file.split('.')[-1] != 'log' and 'chr' in file]  # remove log files
    files = [file for file in files if re.match(filter_term, file)]
    files.sort(key=get_data_ablation_value, reverse=True)
    phenos = set([get_pheno(file) for file in files])

    #
    main_snp_id = get_snp(files[0])

    store = []
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(results_dir, file), sep='\t', error_bad_lines=False)
        df.P = pd.to_numeric(df.P, errors='coerce')
        main_assoc = df[df.ID == main_snp_id]
        log_pval = -np.log10(main_assoc.P.values[0])
        store.append({
            'file': file,
            'phenotype': get_pheno(file),
            'value': get_data_ablation_value(file),
            'id': main_snp_id,
            'log_pval': log_pval,
            'trail': get_trial(file)
        })

    df_results = pd.DataFrame(store)

    # for key, df in df_results.groupby('phenotype'):
    #     plt.plot(df.value, df.log_pval, label=key)

    means = df_results.groupby(['phenotype', 'value']).log_pval.mean()
    stds = df_results.groupby(['phenotype', 'value']).log_pval.std()

    df_results_grouped = means.reset_index()

    df_results_grouped.loc[:, 'stds'] = stds.values

    for key, df in df_results_grouped.groupby('phenotype'):
        df.sort_values('value', inplace=True, ascending=False)
        plt.plot(df.value, df.log_pval, label=key)
        plt.fill_between(df.value, df.log_pval - df.stds, df.log_pval + df.stds, alpha=0.3)

    plt.legend()
    plt.ylabel('-log10 P-value')
    plt.xlabel('Data ablation')
    plt.title('{}'.format(phenotype_name))
    plt.show()
