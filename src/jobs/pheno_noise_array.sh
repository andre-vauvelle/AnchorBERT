#$ -l tmem=16G
#$ -l h_rt=25:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N pheno_noise_array_controls_250.2
#$ -t 1:9
#$ -tc 9

#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
source /share/apps/source_files/cuda/cuda-10.1.source
conda activate

# noise array first element is skipped
NOISE_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#  target_token="'411.2'"
#  target_token="'428.2'"
#  global_params.control_noise="0.1" \

python -O task/pheno.py with \
  target_token=$1 \
  global_params.$2="${NOISE_LEVELS[$SGE_TASK_ID]}"
date
