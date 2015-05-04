#!/usr/bin/env bash

#$ -cwd
#$ -S /bin/bash
#$ -M gkumar6@jhu.edu
#$ -m eas
#$ -l num_proc=1,h_vmem=30g,mem_free=30g,h_rt=120:00:00,gpu=1
#$ -V
#$ -j y -o log/large.log

phraseTable=/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.1.gz
outDir=/export/a04/gkumar/code/custom/rnn/data/1.large

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py \
  -p $phraseTable -o $outDir
