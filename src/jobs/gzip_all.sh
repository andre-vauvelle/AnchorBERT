#!/bin/bash -l

SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit


qsub jobs/gzip.sh $ARGS[0]
