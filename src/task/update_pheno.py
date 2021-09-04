import argparse
import pandas as pd

from task.pheno import update_phenofile, anchor_decorator, apply_anchor, apply_inverse_normal_rank
parser = argparse.ArgumentParser(description='update phenofile with anchor and inverse normal rank, seperate files')


parser.add_argument('phenofile_path', help='full path to phenofile')

args = parser.parse_args()

global_params = {
    'use_code': 'code',  # 'phecode'
    'with_codes': 'all',
    'max_len_seq': 256,
    'inverse_normal_rank_cols': ['logreg_anchor', 'bert_anchor'],  # None to activate for all cols
    'anchor_cols': ['logreg', 'bert', ]
}

phenofile = pd.read_csv(args.phenofile_path, sep='\t')


threshold1_anchor_func = anchor_decorator(apply_anchor, phenofile.threshold1)

phenofile = update_phenofile(threshold1_anchor_func, global_params['anchor_cols'], phenofile=phenofile,
        new_filename=args.phenofile_path[:-4] + '_anchor.tsv',
                             update_colnames='_anchor', drop_cols=True)

update_phenofile(apply_inverse_normal_rank, global_params['inverse_normal_rank_cols'], phenofile=phenofile,
        new_filename=args.phenofile_path[:-4] + '_anchor_inr.tsv',
                             update_colnames='_inr', drop_cols=True)
