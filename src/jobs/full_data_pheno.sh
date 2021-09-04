#!/bin/zsh


SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit

qsub jobs/pheno.sh 'with target_token=250.2'
qsub jobs/pheno.sh 'with target_token=411.2'
qsub jobs/pheno.sh 'with target_token=428.2'
qsub jobs/pheno.sh 'with target_token=714.0|714.1'
qsub jobs/pheno.sh 'with target_token=290.1'