# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# Example from ghaz
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=09:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=8G,tmem=8G
# Find <your_project_id> by running the command "groups"
#$ -N custom_gwas
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors

/share/apps/genomics/plink-2.0/bin/plink2 --1 \
  --pfile /SAN/icsbiobank/UKbiobank_ICS/Projects/GENIUS/GWAS/pgen_format/chr11 \
  --glm cols=+a1freq omit-ref hide-covar \
  --out /SAN/ihibiobank/denaxaslab/andre/pheprob/results/gwas_results/custom \
  --pheno /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/250.2_ca0_co0_anchor.tsv \
  --pheno-name logreg_anchor bert_anchor binomial_r \
  --input-missing-phenotype -999 \
  --covar /SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/covariates.tsv \
  --covar-name sex,age,pca1-pca10 \
  --ci 0.95 \
  --threads "$NSLOTS" \
  --snps rs7928810 rs11602873 rs11024268 rs11603349 rs74046911 rs5210 rs1002226 rs77450170 rs4439492 rs9667947 rs35513985 rs163184 11:17405842_ACT_A rs5222 rs5215 rs61880297 rs2074311 rs7112138 rs2214295 rs77464186 rs12146652 rs34438900 rs35271178 rs7104181 rs67951613 rs2299620 rs5213 rs7484027 rs11606985 rs757110 rs202083422 rs11024271 rs148527516 rs163182 rs2285676 rs11601767 rs10734252 rs2074310 11:17375260_GGCA_G rs234866 rs7109575 rs75780827 rs5219 rs7112030 rs1151517 rs7124355 rs1557765 rs10832776 rs4148646 rs10832778 rs234864 rs2237895 11:72470915_GGTTT_G rs7104177 rs163177 rs199518026 rs10830963 rs2237896 rs10734253 rs2237897 rs12146443 rs10766394 rs142489578 rs2074314 rs2051772