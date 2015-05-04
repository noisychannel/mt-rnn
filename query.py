#!/usr/bin/env python

"""
Queries a pre-trained RNN encoder decoder model to get (average per-token) phrase translation probabilities
"""

import sys
import pickle
import argparse

def prepareQuery(query, sVocab, tVocab, sEmbeddings, tEmbeddings):
  """
  """
  phrasePairs = []
  for line in query:
    line = line.strip().split("|||")
    sPhrase = np.asarray([sVocab.get(w, 0) for w in line[0].strip().split()]).astype('int32')
    tPhrase = np.asarray([tVocab.get(w, 0) for w in line[1].strip().split()]).astype('int32')
    phrasePairs.append((sEmbeddings[sPhrase], tEmbeddings[tPhrase], tPhrase))
  return phrasePairs

parser = argparse.ArgumentParser("Queries a bre-built RNN encoder-decoder model")
parser.add_argument("-m", "--model", dest="modelFile", help="A pre-trained RNN encoder-decoder model (Run train.py to obtain a model file)")
parser.add_argument("-q", "--query", dest="queryFile", help="A file containing queries in the s|||t format")
opts = parser.parse_args()

if opts.modelFile is None or opts.queryFile is None:
  parser.print_help()
  sys.exit(1)

# Parameters that are needed :
[sVocab, tVocab, sEmbedding, tEmbedding, rnn] = [None, None, None, None, None]

# Read parameters from a pickled object
with open(opts.modelFile, "rb") as model:
  [sVocab, tVocab, sEmbedding, tEmbedding, rnn] = pickle.load(model)

# Get the query in a format the RNNED will like
dev = prepareQuery(open(opts.queryFile), sVocab, tVocab, sEmbeddings, tEmbeddings)

for query in dev:
  print rnn.test(query)
