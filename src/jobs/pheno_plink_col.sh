##!/bin/bash -l
# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# Example from ghaz
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=100:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=20G,tmem=20G
# Find <your_project_id> by running the command "groups"
#$ -N gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
# This will create 22 jobs each with trial_n different SGE_TASK_ID which will correspond to trial_n chromosome
#$ -t 1:22
#$ -tc 22

# something like all/pheprob/threshold3.tsv
# or the original data /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/phenos/prev/prev_allcvd_logistic_tabdelim.txt
PHENO_FILE="$1"

PHENO_COL_NAME="$2"

# This the file name which will be used for the results
# /SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/"$OUTPUT_FILE"
OUTPUT_FILE=chr"$SGE_TASK_ID"_"$PHENO_FILE"

# this is the chromosome you want to run the association test on
PFILE=$SGE_TASK_ID



/share/apps/genomics/plink-2.0/bin/plink2 --1 \
  --pfile /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/pgen_format/chr"$PFILE" \
  --glm cols=+a1freq omit-ref hide-covar \
  --out /SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/$OUTPUT_FILE \
  --pheno /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/"$PHENO_FILE" \
  --pheno-name "$PHENO_COL_NAME"\
  --input-missing-phenotype -999 \
  --covar /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/covariates/covariates.tsv \
  --covar-name sex,age,pca1-pca10 \
  --ci 0.95 \
  --threads "$NSLOTS"
#  --$N
