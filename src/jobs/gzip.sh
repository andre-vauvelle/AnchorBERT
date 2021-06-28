#$ -l tmem=16G
#$ -l h_rt=7:0:0
#$ -S /bin/bash
#$ -j y
#$ -N gzip
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs

gzip "$1"
