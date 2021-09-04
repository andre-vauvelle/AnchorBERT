#!/bin/zsh

# first argument must be the phenofile with all subsequent fractions between 0-1


hostname
date
# wait to volumes to attach?
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit

PHENO_FILE=$1

for arg in "${@:2}"
do
  python task/subset_pheno.py -p /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/"$PHENO_FILE" -s "$arg" -c -t 10
done


