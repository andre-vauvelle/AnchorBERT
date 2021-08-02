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
#$ -N gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
# This will create 10 jobs each with trial_n different SGE_TASK_ID which will correspond to trial_n chromosome
#$ -t 1:22
#6, 7, 8, 9, 12, 14, 15, 17, 18, 20, 21
#$ -tc 22

# something like all/pheprob/threshold3.tsv
# or the original data /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/phenos/prev/prev_allcvd_logistic_tabdelim.txt
PHENO_FILE="$1"

# This the file name which will be used for the results
# /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/"$OUTPUT_FILE"
OUTPUT_FILE=chr"$PHENO_FILE"_"$SGE_TASK_ID"

# this is the chromosome you want to run the association test on
PFILE=$SGE_TASK_ID


hostname
echo "USING PFILE: $PFILE PHENO_FILE: $PHENO_FILE"
date
bash /home/vauvelle/pycharm-sftp/pheprob/src/jobs/pheno_plink.sh "$PFILE" "$PHENO_FILE"
date
