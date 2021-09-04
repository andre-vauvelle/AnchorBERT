##!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=100:00:00
#$ -l h_vmem=9G,tmem=9G
#$ -N gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
# This will create 10 jobs each with trial_n different SGE_TASK_ID which will correspond to trial_n chromosome
#$ -t 1:22
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
