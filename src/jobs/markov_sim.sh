#$ -l tmem=16G
#$ -l h_rt=7:0:0
#$ -S /bin/bash
#$ -j y
#$ -N markov

#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs

hostname
date
PROJECT_DIR='/home/vauvelle/pycharm-sftp/pheprob'
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $PROJECT_DIR || exit
source /share/apps/source_files/python/python-3.7.0.source
#source /share/apps/source_files/cuda/cuda-10.1.source
source .myenv/bin/activate
python -W ignore ./src/task/markovian.py
date
