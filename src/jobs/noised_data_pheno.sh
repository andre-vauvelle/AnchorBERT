#$ -l tmem=16G
#$ -l h_rt=10:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N noised_data_pheno
#$ -t 1:45
#$ -tc 45
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit

CONFIG_PATH=$1
ROW=$SGE_TASK_ID
LINE=$(sed -n $((ROW + 1))'{p;q}' "$CONFIG_PATH")
IFS=',' read -a ARGS <<<"$LINE"
WITH='with target_token='"${ARGS[0]}"' global_params.case_noise='"${ARGS[1]}"' bert_config.train_params.epochs=10'
echo "$WITH"

source /share/apps/source_files/cuda/cuda-10.1.source
conda activate
python -O task/pheno.py $WITH
date
