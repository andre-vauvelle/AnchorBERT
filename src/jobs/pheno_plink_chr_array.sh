##!/bin/bash -l
# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# Example from ghaz
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=100:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=15G,tmem=15G
# Find <your_project_id> by running the command "groups"
#$ -N gwas_subset
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
# This will create 10 jobs each with trial_n different SGE_TASK_ID which will correspond to trial_n chromosome
#$ -t 1:9
#6, 7, 8, 9, 12, 14, 15, 17, 18, 20, 21
#$ -tc 9

#PHENO_FILE_SEARCH="$1"
PHENO_FILE_SEARCH="bert_binomial_r_logreg_threshold1_threshold2_threshold3_250.2_0*"
cd /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/ || exit
PHENO_FILE="$(find . -name "$PHENO_FILE_SEARCH" | sed -n "$SGE_TASK_ID"p | cut -c 3-)" || {
  echo 'Could not find files'
  exit 1
}

# this is the chromosome you want to run the association test on
PFILE="9"

# /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/"$OUTPUT_FILE"
OUTPUT_FILE=chr"$CHRFILE"_"$PHENO_FILE"

hostname
echo "USING PFILE: $PFILE PHENO_FILE: $PHENO_FILE"
date
bash /home/vauvelle/pycharm-sftp/pheprob/src/jobs/pheno_plink.sh "$PFILE" "$PHENO_FILE"
date
