#!/usr/bin/env python

"""
For internal use only

Upgrades an old RNNED model (where the entire object gets pickled) to a new model format (only write params)
"""

import sys
import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser("Upgrades an RNNED model")
parser.add_argument("-m", "--model", dest="modelFile", help="A pre-trained RNN encoder-decoder model (Run train.py to obtain a model file)")
opts = parser.parse_args()

if opts.modelFile is None:
  parser.print_help()
  sys.exit(1)

# Parameters that are needed :
[sVocab, tVocab, sEmbedding, tEmbedding, rnn] = [None, None, None, None, None]

oldModelFile = opts.modelFile
oldModelFile = opts.modelFile + ".old"
os.system("mv " + opts.modelFile + " " + oldModelFile)

# Read parameters from a pickled object
# Yield your secrets!
with open(oldModelFile, "rb") as model:
  [sVocab, tVocab, sEmbeddings, tEmbeddings, rnn] = pickle.load(model)

lParameters = [sVocab, tVocab, sEmbeddings, tEmbeddings]
rParameters = [p.get_value() for p in rnn.params]

with open(opts.modelFile, "wb") as m:
  pickle.dump([[lParameters], [rParameters]], m)
