#$ -l tmem=6G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -N gzip
#$ -o /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs
#$ -e /home/vauvelle/pycharm-sftp/pheprob/src/jobs/logs/errors
#$ -t 1:24
#$ -tc 24

CONFIG_PATH=$1
ROW=$SGE_TASK_ID
#ROW=1
LINE=$(sed -n $((ROW))'{p;q}' $CONFIG_PATH)

IFS=',' read -a ARGS <<<"$LINE"

ZIPIT=${ARGS[0]}

echo "$ZIPIT"

gzip -f "$ZIPIT"
