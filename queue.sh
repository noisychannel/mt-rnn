#!/usr/bin/env bash

#$ -cwd
#$ -S /bin/bash
#$ -M gkumar6@jhu.edu
#$ -m eas
#$ -l num_proc=1,h_vmem=30g,mem_free=30g,h_rt=120:00:00,gpu=1
#$ -V
#$ -j y -o log/small.log

echo "Starting job on : " `hostname`
echo "Started at : " `date`

phraseTable=/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.small.1.gz
#phraseTable=/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.1.gz
outDir=/export/a04/gkumar/code/custom/rnn/data/1.small
#outDir=/export/a04/gkumar/code/custom/rnn/data/1.large

# Keep compiled files at the place where the job runs
# Backoff to the default dir
defaultCompileDir=/export/`hostname`/gkumar/tmp
if [ -d $defaultCompileDir ]; then
  compileDir=$defaultCompileDir/gk.$RANDOM
else
  compileDir=/export/a04/gkumar/tmp/gk.$RANDOM
fi

THEANO_FLAGS=compiledir=${compileDir},mode=FAST_RUN,device=gpu,floatX=float32 python train.py \
  -p ${phraseTable} -o ${outDir}

echo "Finished job at " `date`
