# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# Example from ghaz
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=09:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=8G,tmem=8G
# Find <your_project_id> by running the command "groups"
#$ -N gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
#$ -t 1:42
#$ -tc 42

# This the file name which will be used for the results
# /SAN/ihibiobank/denaxaslab/andre/pheprob/gwas_results/"$OUTPUT_FILE"


CONFIG_PATH=$1
ROW=$SGE_TASK_ID
#ROW=1
LINE=$(sed -n $((ROW + 1))'{p;q}' $CONFIG_PATH)

IFS=',' read -a ARGS <<< "$LINE"

PFILE=chr${ARGS[0]}
PHENO_DIR=${ARGS[1]}
SNPS=${ARGS[2]}
N_TRIALS=${ARGS[3]}


OUTPUT_FILE="$PFILE"_"$PHENO_DIR"

echo "$ARGS"

sleep "$(shuf -i 1-30 -n1)"

hostname
date
# wait to volumes to attach?
for ((i=0;i<N_TRIALS;i++)); do
  echo "$i"
  OUTPUT_FILE="$PFILE"_"$PHENO_DIR"_"$i"
  /share/apps/genomics/plink-2.0/bin/plink2 --1 \
    --pfile /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/pgen_format/"$PFILE" \
    --glm cols=+a1freq omit-ref hide-covar \
    --out /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/trials/"$OUTPUT_FILE" \
    --pheno /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/"$PHENO_DIR"/"$i".tsv \
    --input-missing-phenotype -999 \
    --covar /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/covariates/covariates.tsv \
    --covar-name sex,age,pca1-pca10 \
    --ci 0.95 \
    --snps $SNPS
#    --threads "$NSLOTS" \

done