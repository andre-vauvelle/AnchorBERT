#$ -l tmem=16G
#$ -l h_rt=50:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N gpu_workers
#$ -t 1-20
#$ -tc 4

#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source /share/apps/source_files/cuda/cuda-10.1.source
conda activate
echo "Pulling any jobs with status 0"
hyperopt-mongo-worker --mongo=bigtop:27017/hyperopt --poll-interval=0.1 --max-consecutive-failures=5
date
