##!/bin/bash -l
# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# Example from ghaz
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=90:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=15G,tmem=15G
# Find <your_project_id> by running the command "groups"
#$ -N gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors

# This the file name which will be used for the results
# /SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/"$OUTPUT_FILE"
OUTPUT_FILE=chr"$1"_"$2"_"$3"

# this is the chromosome you want to run the association test on
PFILE=chr"$1"

# something like all/pheprob/threshold3.tsv
# or the original data /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/phenos/prev/prev_allcvd_logistic_tabdelim.txt
PHENO_FILE="$2"

SNPS="$3"

hostname
date
# wait to volumes to attach?
/share/apps/genomics/plink-2.0/bin/plink2 --1 \
  --pfile /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/pgen_format/"$PFILE" \
  --glm cols=+a1freq omit-ref hide-covar \
  --out /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/"$OUTPUT_FILE" \
  --pheno /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/"$PHENO_FILE" \
  --input-missing-phenotype -999 \
  --covar /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/covariates.tsv \
  --covar-name sex,age,pca1-pca10 \
  --ci 0.95 \
  --threads "$NSLOTS" \
  --snps "$SNPS"

#qstat -j $JOB_ID