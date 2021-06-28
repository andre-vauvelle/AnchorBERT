#$ -l tmem=16G
#$ -l h_rt=50:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N pheno

#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source /share/apps/source_files/cuda/cuda-10.1.source
conda activate
python -O task/pheno.py
date