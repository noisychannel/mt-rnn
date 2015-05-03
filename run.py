#!/usr/bin/env python

# TODO:
# 1. Get the phrase table and extract all phrases (Use the version from Phrase translation)
# 2. Get vocabulary : Keep the top 10k words and map the rest to UNK
# 3. Apply the UNK changes to the phrase table
# 4. Get the embeddings for the modified corpus
# 5. Shuffle the phrase pairs from the phrase table and split into train and dev sets (80/20)
# 6. Form mini batches from the training data
# 7. What is the structure of a phrase vector ? What is a good way of selecting these ? Draw inspiration from the sentence selection in the Theano example
# 8. Only apply averaged updates after a mini-batch is complete
# 9. Run for several epochs. The most successful epoch is the one which maximin
# 10. Retain best model in the form of a pickle dump so that it can be loaded by the feature function in moses

import os
import sys
import gzip
import time
import codecs
import random
import operator
import argparse
import numpy as np
from collections import defaultdict
import rnn_encoder_decoder as rnned


def readWordVectors(vectorBin, vocab, dim):
  """
  Create a WordVectors class based on a word2vec binary file

  Parameters
  ----------

  """
  # First read vectors into a temporary hash
  vectorHash = defaultdict()
  with open(vectorBin) as fin:
    header = fin.readline()
    vocab_size, vector_size = map(int, header.split())

    assert vector_size == dim

    binary_len = np.dtype(np.float32).itemsize * vector_size
    for line_number in xrange(vocab_size):
      # mixed text and binary: read text first, then binary
      word = ''
      while True:
        ch = fin.read(1)
        if ch == ' ':
          break
        word += ch

      vector = np.fromstring(fin.read(binary_len), np.float32)
      vectorHash[word.decode('utf8')] = vector
      fin.read(1)  # newline

  # Now create the embedding matrix
  # TODO:What is the shape
  embeddings = np.empty((len(vocab), dim), dtype=np.float32)
  # Embedding for the unknown symbol
  unk = np.ones((dim))
  # We don't want to count the explicit UNK as an unknown
  unkCount = -1
  for i in range(len(vocab)):
    if vocab[i] not in vectorHash:
      unkCount += 1
    embeddings[i] = vectorHash.get(vocab[i], unk)

  del vectorHash
  return unkCount, embeddings


def parseCorpus(iFile, pruneThreshold):
  """
  """
  freq = defaultdict()
  for line in iFile:
    words = line.strip().split()
    for word in words:
      freq[word] = freq.get(word, 0) + 1

  # Sort the frequencies
  wordCounts = reduce(lambda x, y: x + y, freq.values())
  freqSort = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
  # Prune the vocab
  # TODO: Change this to a parameter
  freqSort = freqSort[:pruneThreshold]
  prunedWordCounts = reduce(lambda x, y: x + y, [x[1] for x in freqSort])
  vocab = defaultdict()
  rVocab = defaultdict()
  vocab["UNK"] = 0
  rVocab[0] = "UNK"
  vocabID = 0
  for item in freqSort:
    vocabID += 1
    vocab[item[0]] = vocabID
    rVocab[vocabID] = item[0]

  return float(prunedWordCounts)/wordCounts, vocab, rVocab


def minibatch(l, bs):
  """
  Yield batches
  """
  for i in xrange(0, len(l), bs):
    yield l[i:i+bs]


def processBatch(batch):
  """
  Processes a batch to the format required by the RNNED functions
  """
  X = np.empty((len(batch), batch[0][0].shape[0], batch[0][0].shape[0]))
  Y = np.empty((len(batch), batch[0][1].shape[0], batch[0][1].shape[0]))
  Y_IDX = np.empty((len(batch), batch[0][2].shape[0]))
  for i, (x, y, y_idx) in enumerate(batch):
    X[i] = x
    Y[i] = y
    Y_IDX[i] = y_idx

  return X, Y, Y_IDX


def getPhrasePairs(tTable, sVocab, tVocab, sEmbeddings, tEmbeddings):
  """
  """
  phrasePairs = []
  for line in tTable:
    line = line.strip().split("|||")
    sPhrase = np.asarray([sVocab.get(w, 0) for w in line[0].strip().split()]).astype('int32')
    tPhrase = np.asarray([tVocab.get(w, 0) for w in line[1].strip().split()]).astype('int32')
    # Don't include phrases that contain only OOVs
    if np.sum(sPhrase) == 0 or np.sum(tPhrase) == 0:
      continue
    phrasePairs.append((sEmbeddings[sPhrase], tEmbeddings[tPhrase], tPhrase))

  return phrasePairs


def shuffle(l, seed):
  """
  """
  random.seed(seed)
  random.shuffle(l)


def getPartitions(phrasePairs, seed):
  """
  """
  shuffle(phrasePairs, seed)
  # 80/20 partition for train/dev
  return phrasePairs[:int(0.8 * len(phrasePairs))], phrasePairs[int(0.8 * len(phrasePairs)):]


parser = argparse.ArgumentParser("Runs the RNN encoder-decoder training procedure for machine translation")
parser.add_argument("-p", "--phrase-table", dest="phraseTable",
    default="/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.small.1.gz", help="The location of the phrase table")
    #default="/export/a04/gkumar/experiments/MT-JHU/1/model/phrase-table.1.gz", help="The location of the phrase table")
parser.add_argument("-f", "--source", dest="sFile",
    default="/export/a04/gkumar/corpora/fishcall/kaldi_fishcall_output/SAT/ldc/processed/fisher_train.tok.lc.clean.es",
    help="The training text for the foreign (target) language")
parser.add_argument("-e", "--target", dest="tFile",
    default="/export/a04/gkumar/corpora/fishcall/kaldi_fishcall_output/SAT/ldc/processed/fisher_train.tok.lc.clean.en",
    help="The training text for the english (source) language")
parser.add_argument("-s", "--source-emb", dest="sEmbeddings",
    default="/export/a04/gkumar/code/custom/brae/tools/word2vec/fisher_es.vectors.50.sg.bin", help="Source embeddings obtained from word2vec")
parser.add_argument("-t", "--target-emb", dest="tEmbeddings",
    default="/export/a04/gkumar/code/custom/brae/tools/word2vec/fisher_en.vectors.50.sg.bin", help="Target embeddings obtained from word2vec")
opts = parser.parse_args()

# Hyperparameters
s = {
  'lr': 0.627, # The learning rate
  #'bs':1000, # number of backprops through time steps
  'bs':1000, # number of backprops through time steps
  'nhidden':100, # Size of the hidden layer
  'seed':324, # Seed for the random number generator
  'emb_dimension':50, # The dimension of the embedding
  'nepochs':10, # The number of epochs that training is to run for
  'prune_t':5000 # The frequency threshold for histogram pruning of the vocab
}

# First process the training dataset and get the source and target vocabulary
start = time.time()
sCoverage, s2idx, idx2s = parseCorpus(codecs.open(opts.sFile, encoding="utf8"), s['prune_t'])
tCoverage, t2idx, idx2t = parseCorpus(codecs.open(opts.tFile, encoding="utf8"), s['prune_t'])
print "***", sCoverage*100, "% of the source corpus covered by the pruned vocabulary"
print "***", tCoverage*100, "% of the target corpus covered by the pruned vocabulary "
print "--- Done creating vocabularies : ", time.time() - start, "s"

# Get embeddings for the source and the target phrase pairs
start = time.time()
sUnkCount, sEmbeddings = readWordVectors(opts.sEmbeddings, idx2s, s['emb_dimension'])
tUnkCount, tEmbeddings = readWordVectors(opts.tEmbeddings, idx2t, s['emb_dimension'])
print "***", sUnkCount, " source types were not seen in the embeddings"
print "***", tUnkCount, " target types were not seen in the embeddings"
print "--- Done reading embeddings for source and target : ", time.time() - start, "s"

# Now, read the phrase table and get the phrase pairs for training
start = time.time()
phraseTable = gzip.open(opts.phraseTable)
phrasePairs = getPhrasePairs(phraseTable, s2idx, t2idx, sEmbeddings, tEmbeddings)
print "--- Done reading phrase pairs from the phrase table : ", time.time() - start, "s"

# Create the training and the dev partitions
train, dev = getPartitions(phrasePairs, s['seed'])

tVocSize = len(t2idx)
nTrainExamples = len(train)

start = time.time()
rnn = rnned.RNNED(nh=s['nhidden'], nc=tVocSize, de=s['emb_dimension'])
print "--- Done compiling theano functions : ", time.time() - start, "s"

for e in xrange(s['nepochs']):
  shuffle(train, s['seed'])
  s['ce'] = e
  tic = time.time()
  for i, batch in enumerate(minibatch(train, s['bs'])):
    rnn.train(batch, s['lr'])

  print '[learning] epoch', e,  '>> completed in', time.time() - tic, '(sec) <<'
  sys.stdout.flush()

  rnn.test(dev)
