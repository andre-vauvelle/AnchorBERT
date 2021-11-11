#!/bin/zsh

SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit
# With hyperpameters found from tuning
qsub jobs/pheno.sh 'with target_token=250.2 bert_config.model_config.hidden_size=360 bert_config.model_config.intermediate_size=516 bert_config.optim_config.lr=0.0001 bert_config.model_config.num_attention_heads=12 bert_config.model_config.num_hidden_layers=6'
qsub jobs/pheno.sh 'with target_token=411.2 bert_config.model_config.hidden_size=360 bert_config.model_config.intermediate_size=516 bert_config.optim_config.lr=0.0001 bert_config.model_config.num_attention_heads=12 bert_config.model_config.num_hidden_layers=2'
qsub jobs/pheno.sh 'with target_token=428.2 bert_config.model_config.hidden_size=360 bert_config.model_config.intermediate_size=256 bert_config.optim_config.lr=0.0001 bert_config.model_config.num_attention_heads=12 bert_config.model_config.num_hidden_layers=10'
qsub jobs/pheno.sh 'with target_token=714.0|714.1 bert_config.model_config.hidden_size=360 bert_config.model_config.intermediate_size=256 bert_config.optim_config.lr=0.0001 bert_config.model_config.num_attention_heads=12 bert_config.model_config.num_hidden_layers=6'
qsub jobs/pheno.sh 'with target_token=290.1 bert_config.model_config.hidden_size=360 bert_config.model_config.intermediate_size=256 bert_config.optim_config.lr=0.0001 bert_config.model_config.num_attention_heads=12 bert_config.model_config.num_hidden_layers=6'

#To do 10 times for confidence intervals
#for i in `seq 10`; do bash jobs/full_data_pheno.sh; done
