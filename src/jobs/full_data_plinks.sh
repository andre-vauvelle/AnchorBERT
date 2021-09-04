#!/bin/zsh

SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit

qsub jobs/pheno_plink_array.sh 428.2_ca0_co0_anchor.tsv
qsub jobs/pheno_plink_array.sh '714.0|714.1_ca0_co0_anchor.tsv'
qsub jobs/pheno_plink_array.sh 411.2_ca0_co0_anchor.tsv
qsub jobs/pheno_plink_array.sh 250.2_ca0_co0_anchor.tsv
qsub jobs/pheno_plink_array.sh 290.1_ca0_co0_anchor.tsv