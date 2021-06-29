import pandas as pd
import os

from sklearn.metrics import jaccard_score
from tqdm import tqdm

from definitions import LOCAL_RESULTS_DIR
import numpy as np


def get_signficant_loci(search='250', genome_wide_assoc=5e-8):
    files = os.listdir(os.path.join(LOCAL_RESULTS_DIR, 'data'))
    files = [f for f in files if search in f]

    store = []
    for file in tqdm(files, desc='Getting significant loci for: {}'.format(search)):
        # file = files[-6]
        filepath = os.path.join(LOCAL_RESULTS_DIR, 'data', file)
        try:
            df = pd.read_csv(filepath, sep='\t')
        except Exception as e:
            print('Could not read:{}'.format(filepath))
            raise e
        df.loc[:, 'p_log'] = -np.log10(df.P)
        significant_loci = df[df.P <= genome_wide_assoc].copy()
        significant_loci.loc[:, 'phenotype_full'] = file.split('.')[0]
        significant_loci.loc[:, 'method'] = '_'.join(file.split('_')[:-1])
        significant_loci.loc[:, 'phenotype'] = file.split('.')[0].split('_')[-1]
        store.append(significant_loci)

    df_significant = pd.concat(store, axis=0)
    return df_significant


def get_loci(df, method='pos'):
    if method == 'pos':
        loci = df.CHROM.astype(str) + '_' + df.POS.astype(str)
        return loci.values
    if method == 'rsid':
        loci = df.ID
        return loci.values
    else:
        return None


def jaccard_index(a, b):
    intersection = len(set.intersection(set(a), set(b)))
    union = len(set.union(set(a), set(b)))
    return float(intersection / union)


def get_results(df_significant, loci_method='pos'):
    results_store = []
    phenotype_groups = df_significant.groupby('phenotype')
    for pheno_name, df_pheno in phenotype_groups:
        df_baseline = df_pheno[df_pheno.method == 'threshold1']
        baseline_loci = get_loci(df_baseline, method=loci_method)
        method_groups = df_pheno.groupby('method')
        for method_name, df_method in method_groups:
            method_loci = get_loci(df_method, method=loci_method)

            results_store.append({
                'pheno': pheno_name,
                'name': method_name,
                'jaccard_index': jaccard_index(baseline_loci, method_loci),
                'intersect': len(set.intersection(set(baseline_loci), set(method_loci))),
                'new': len(set(method_loci) - set(baseline_loci)),
                'missing': len(set(baseline_loci) - set(method_loci)),
                'missing_p': len(set(baseline_loci) - set(method_loci)) / len(set(baseline_loci)),
            })

    df_results = pd.DataFrame(results_store)
    return df_results


if __name__ == 'main':
    df_significant = get_signficant_loci(search='')
    df_results = get_results(df_significant, loci_method='pos')
    df_results = get_results(df_significant, loci_method='rsid')
