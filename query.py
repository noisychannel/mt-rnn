#!/usr/bin/env python

"""
Queries a pre-trained RNN encoder decoder model to get (average per-token) phrase translation probabilities
"""

import sys
import math
import gzip
import numpy as np
import time
import pickle
import argparse
import rnn_encoder_decoder as rnned

def getPhrasePairs(query, sVocab, tVocab, sEmbeddings, tEmbeddings):
  """
  """
  phrasePairs = []
  rawPairs = []
  for line in query:
    line = line.strip().split("|||")
    rawPairs.append(line)
    sPhrase = np.asarray([sVocab.get(w, 0) for w in line[0].strip().split()]).astype('int32')
    tPhrase = np.asarray([tVocab.get(w, 0) for w in line[1].strip().split()]).astype('int32')
    phrasePairs.append((sEmbeddings[sPhrase], tEmbeddings[tPhrase], tPhrase))
  return phrasePairs, rawPairs


def minibatch(l, bs):
  """
  Yield batches for mini-batch SGD

  Parameters:
    l : The list of training examples
    bs : The batch size

  Returns:
    Iterator over batches
  """
  for i in xrange(0, len(l), bs):
    yield l[i:i+bs]


parser = argparse.ArgumentParser("Queries a pre-built RNN encoder-decoder model")
parser.add_argument("-m", "--model", dest="modelFile", help="A pre-trained RNN encoder-decoder model (Run train.py to obtain a model file)")
parser.add_argument("-p", "--phrase-table", dest="phraseTable", help="A moses format phrase table")
parser.add_argument("-o", "--output", dest="outputFile", help="The output phrase table")
opts = parser.parse_args()

if opts.modelFile is None or opts.phraseTable is None or opts.outputFile is None:
  parser.print_help()
  sys.exit(1)

start = time.time()

lParameters = None
rParameters = None
# Read parameters from a pickled object
with open(opts.modelFile, "rb") as model:
  [[lParameters], [rParameters]] = pickle.load(model)

[sVocab, tVocab, sEmbeddings, tEmbeddings] = lParameters
print "--- Done loading pickled parameters : ", time.time() - start, "s"

start = time.time()
# Infer parameters from the hidden variables
nh = rParameters[0].shape[0]
nc = rParameters[-1].shape[0]
de = rParameters[-1].shape[1]
rnn = rnned.RNNED(nh, nc, de, rParameters)
print "--- Done creating RNNED object : ", time.time() - start, "s"

start = time.time()
phraseTable = gzip.open(opts.phraseTable)
phrasePairs, rawPhrases = getPhrasePairs(phraseTable, sVocab, tVocab, sEmbeddings, tEmbeddings)
print "--- Done reading phrase pairs from the phrase table : ", time.time() - start, "s"
# Get the query in a format the RNNED will like
#dev = prepareQuery(open(opts.queryFile), sVocab, tVocab, sEmbeddings, tEmbeddings)

outputTT = open(opts.outputFile, "w+")

bs = 10000
tic = time.time()
all_scores = []
for i, batch in enumerate(minibatch(phrasePairs, bs)):
  dev_nlls = rnn.test(batch)
  all_scores += list(dev_nlls)
  print '[BATCH', i, "] decoded in", time.time() - tic, '(sec)'
  print '[AVG-NLL]', np.mean(dev_nlls)
  sys.stdout.flush()
  tic = time.time()

for line,score in zip(rawPhrases, all_scores):
  line[2] += str(round(math.exp(-1. * score),8))
  outputTT.write(" ||| ".join([x.strip() for x in line]) + "\n")

outputTT.close()
