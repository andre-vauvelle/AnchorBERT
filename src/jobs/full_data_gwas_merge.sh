#!/bin/zsh


SOURCE_DIR='/home/vauvelle/pycharm-sftp/pheprob/src/'
cd $SOURCE_DIR || exit

PHENO=250.2_ca0_co0_anchor.tsv.
qsub jobs/merge_plinks.sh "$PHENO"binomial_r "$PHENO"binomial_r
qsub jobs/merge_plinks.sh "$PHENO"logreg_anchor "$PHENO"logreg_anchor
qsub jobs/merge_plinks.sh "$PHENO"bert_anchor "$PHENO"bert_anchor
qsub jobs/merge_plinks.sh "$PHENO"threshold1 "$PHENO"threshold1
qsub jobs/merge_plinks.sh "$PHENO"threshold2 "$PHENO"threshold2
qsub jobs/merge_plinks.sh "$PHENO"threshold3 "$PHENO"threshold3

PHENO=411.2_ca0_co0_anchor.tsv.
qsub jobs/merge_plinks.sh "$PHENO"binomial_r "$PHENO"binomial_r
qsub jobs/merge_plinks.sh "$PHENO"logreg_anchor "$PHENO"logreg_anchor
qsub jobs/merge_plinks.sh "$PHENO"bert_anchor "$PHENO"bert_anchor
qsub jobs/merge_plinks.sh "$PHENO"threshold1 "$PHENO"threshold1
qsub jobs/merge_plinks.sh "$PHENO"threshold2 "$PHENO"threshold2
qsub jobs/merge_plinks.sh "$PHENO"threshold3 "$PHENO"threshold3

PHENO=428.2_ca0_co0_anchor.tsv.
qsub jobs/merge_plinks.sh "$PHENO"binomial_r "$PHENO"binomial_r
qsub jobs/merge_plinks.sh "$PHENO"logreg_anchor "$PHENO"logreg_anchor
qsub jobs/merge_plinks.sh "$PHENO"bert_anchor "$PHENO"bert_anchor
qsub jobs/merge_plinks.sh "$PHENO"threshold1 "$PHENO"threshold1
qsub jobs/merge_plinks.sh "$PHENO"threshold2 "$PHENO"threshold2
qsub jobs/merge_plinks.sh "$PHENO"threshold3 "$PHENO"threshold3


PHENO='714.0|714.1_ca0_co0_anchor.tsv.'
qsub jobs/merge_plinks.sh "$PHENO"binomial_r "$PHENO"binomial_r
qsub jobs/merge_plinks.sh "$PHENO"logreg_anchor "$PHENO"logreg_anchor
qsub jobs/merge_plinks.sh "$PHENO"bert_anchor "$PHENO"bert_anchor
qsub jobs/merge_plinks.sh "$PHENO"threshold1 "$PHENO"threshold1
qsub jobs/merge_plinks.sh "$PHENO"threshold2 "$PHENO"threshold2
qsub jobs/merge_plinks.sh "$PHENO"threshold3 "$PHENO"threshold3


PHENO=290.1_ca0_co0_anchor.tsv.
qsub jobs/merge_plinks.sh "$PHENO"binomial_r "$PHENO"binomial_r
qsub jobs/merge_plinks.sh "$PHENO"logreg_anchor "$PHENO"logreg_anchor
qsub jobs/merge_plinks.sh "$PHENO"bert_anchor "$PHENO"bert_anchor
qsub jobs/merge_plinks.sh "$PHENO"threshold1 "$PHENO"threshold1
qsub jobs/merge_plinks.sh "$PHENO"threshold2 "$PHENO"threshold2
qsub jobs/merge_plinks.sh "$PHENO"threshold3 "$PHENO"threshold3
