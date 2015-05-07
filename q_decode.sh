#!/usr/bin/env bash

#$ -cwd
#$ -S /bin/bash
#$ -M gkumar6@jhu.edu
#$ -m eas
#$ -l num_proc=1,h_vmem=30g,mem_free=30g,h_rt=120:00:00,gpu=1
#$ -V
#$ -j y -o log/decode.medium.2.log

echo "Starting job on : " `hostname`
echo "Started at : " `date`

phraseTT=/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.1.gz

model=data/1.medium.2/best.mdl
outTT=data/1.medium.2/phrase-table.1

# Keep compiled files at the place where the job runs
# Backoff to the default dir
defaultCompileDir=/export/`hostname`/gkumar/tmp
if [ -d $defaultCompileDir ]; then
  compileDir=$defaultCompileDir/gk.$RANDOM
else
  compileDir=/export/a04/gkumar/tmp/gk.$RANDOM
fi

THEANO_FLAGS=compiledir=${compileDir},mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=False,nvcc.fastmath=True \
  python query.py -m ${model} -p ${phraseTT} -o ${outTT}

gzip ${outTT}

echo "Finished job at " `date`
