#$ -l tmem=16G
#$ -l h_rt=10:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N anchor_bert_hypergrid
#$ -t 1:15
#$ -tc 5
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors

hostname
date
SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR
cd $SOURCE_DIR || exit
#    params = {
#        'lr': [1e-5, 1e-4, 1e-3],
#        'hidden_size': [100, 200, 300],
#        'num_hidden_layers': [2, 6, 10],
#        'num_attention_heads': [4, 12, 14],  # number of attention heads
#        'intermediate_size': [128, 256, 516]
#    }

TARGET=$1

CONFIG_PATH='/home/vauvelle/pycharm-sftp/pheprob/src/jobs/configs/anchorbert_paramgrid-16.csv'
ROW=$SGE_TASK_ID
LINE=$(sed -n $((ROW + 1))'{p;q}' "$CONFIG_PATH")
IFS=',' read -a ARGS <<<"$LINE"

WITH='with target_token='"$TARGET"' bert_config.model_config.hidden_size='${ARGS[0]}' bert_config.model_config.intermediate_size='${ARGS[1]}' bert_config.optim_config.lr='${ARGS[2]}' bert_config.model_config.num_attention_heads='${ARGS[3]}' bert_config.model_config.num_hidden_layers='${ARGS[4]}
echo "$WITH"

source /share/apps/source_files/cuda/cuda-10.1.source
conda activate
python -O task/AnchorBERT.py $WITH
date
