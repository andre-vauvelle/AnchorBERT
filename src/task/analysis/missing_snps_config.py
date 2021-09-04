import pandas as pd

if __name__ == '__main__':
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

    store = []
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

        gwas_s = [big_gwas_thr1, big_gwas_thr2, big_gwas_thr3, big_gwas_bert, big_gwas_logreg, big_gwas_binomial]
        gwas_s_na = ['big_gwas_thr1', 'big_gwas_thr2', 'big_gwas_thr3', 'big_gwas_bert', 'big_gwas_logreg',
                     'big_gwas_binomial']
        for g, n in zip(gwas_s, gwas_s_na):
            store.append({'code': code, n: set(range(1, 23)) - set(g.CHROM)})




    df = pd.DataFrame(store)
    df = df.melt(id_vars='code')
    df.dropna(inplace=True)
    chroms_to_rerun = df.value.explode().dropna()
    df_rerun = df.loc[:, ['code', 'variable']].merge(chroms_to_rerun.to_frame(), left_index=True, right_index=True)

    code2file = {
        '250.2': '250.2_ca0_co0_anchor.tsv',
        '428.2': '428.2_ca0_co0_anchor.tsv',
        '290.1': '290.1_ca0_co0_anchor.tsv',
        '714.0|714.1': '714.0|714.1_ca0_co0_anchor.tsv',
        '411.2': '411.2_ca0_co0_anchor.tsv'
    }
    path2col = {'big_gwas_bert': 'bert_anchor',
                'big_gwas_logreg': 'logreg_anchor',
                'big_gwas_binomial': 'binomial_r',
                'big_gwas_thr1': 'threshold1',
                'big_gwas_thr2': 'threshold2',
                'big_gwas_thr3': 'threshold3'}

    df_rerun.loc[:, 'phenofile'] = df_rerun.code.map(code2file)
    df_rerun.loc[:, 'col'] = df_rerun.variable.map(path2col)

    df_rerun.loc[:, ['phenofile', 'col', 'value']]
    df_config = df_rerun.groupby(['phenofile', 'value']).col.apply(lambda x: ' '.join(x)).reset_index()
    df_config.to_csv('/home/vauvelle/pycharm-sftp/pheprob/src/jobs/configs/missing_full_data_chr-{}.csv'.format(df_config.shape[0]),
              index=False)