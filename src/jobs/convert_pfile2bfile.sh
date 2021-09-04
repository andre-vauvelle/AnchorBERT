##!/bin/bash -l
# Batch script to run trial_n serial job on Legion with the upgraded # software stack under SGE.
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=90:00:00
# 3. Request 20 gigabyte of RAM
#$ -l h_vmem=15G,tmem=15G
# Find <your_project_id> by running the command "groups"
#$ -N convertp2b
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors


/share/apps/genomics/plink-1.9/plink