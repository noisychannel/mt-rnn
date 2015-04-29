#!/usr/bin/env python

import cPickle
import random
import numpy
import exrnn
import time
import sys
import os
import subprocess
from accuracy import conlleval

def contextwin(l, win):
  """
  win : int corresponding to the size of the window
  given a list of indices composing a sentence

  l : array containing the word indices
  """
  assert (win % 2) == 1
  assert win >= 1
  l = list(l)

  # Think of the window surrounding each word, a win//2 padding is needed for this to work
  lpadded = win // 2 * [-1] + l + win // 2 * [-1]
  out = [lpadded[i:(i + win)] for i in range(len(l))]

  assert len(l) == len(out)
  return out


def minibatch(l, bs):
  """
  Return a list of mini batches of indices
  equal to the size of bs
  """
  out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
  out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
  assert len(l) == len(out)
  return out


def shuffle(lol, seed):
  """
  Shuffle a list of list inplace
  lol : list of lists
  seed : a random seed
  """
  for l in lol:
    random.seed(seed)
    random.shuffle(l)


# Hyperparameters
s = {'fold': None, # Not used
    'lr': 0.627, # The learning rate
    'decay':False, # decay the learning rate if improvement stops
    'win':7, # the size of the context window
    'bs':9, # number of backprops through time steps
    'nhidden':100, # Size of the hidden layer
    'seed':324, # Seed for the random number generator
    'emb_dimension':100, # The dimension of the embedding
    'nepochs':50 # The number of epochs that training is to run for
    }

#folder = os.path.basename(__file__).split('.')[0]
folder = 'exp'

# Initializes the training and test datasets and the corresponding vocabularies
# Read the train and the test data
train, test, dicts = cPickle.load(open("data/atis.pkl"))

# Process dictionaries for easy use
words2idx = dicts["words2idx"]
labels2idx = dicts["labels2idx"]
idx2words = {v:k for k,v in words2idx.iteritems()}
idx2labels = {v:k for k,v in labels2idx.iteritems()}

# TODO: What is this ne stuff?
train_x, train_ne, train_y = train
test_x, test_ne, test_y = test
# Visualize some data
print map(lambda x: idx2words[x], train[0][0])
print map(lambda x: idx2labels[x], train[2][0])

vocSize = len(set(reduce(lambda x,y: list(x) + list(y),train_x + test_x)))
nClasses = len(set(reduce(lambda x,y: list(x) + list(y),train_y + test_y)))
nSentences = len(train_x)

numpy.random.seed(s['seed'])
random.seed(s['seed'])

rnn = exrnn.RNNSLU(nh=s['nhidden'], nc=nClasses, ne=vocSize, de=s['emb_dimension'], cs=s['win'])

# Train
bestF1 = -numpy.inf
s['clr'] = s['lr']

for e in xrange(s['nepochs']):
  # Shuffle
  shuffle([train_x, train_ne, train_y], s['seed'])
  s['ce'] = e
  tic = time.time()
  for i in xrange(nSentences):
    cwords = contextwin(train_x[i], s['win'])
    words = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cwords, s['bs']))
    labels = train_y[i]
    for word_batch, label_last_word in zip(words, labels):
      rnn.sentence_train(word_batch, label_last_word, s['clr'])
      rnn.normalize()
    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nSentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
    sys.stdout.flush()

  predictions_test = [map(lambda x: idx2labels[x], \
      rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) \
        for x in test_x]
  groundtruth_test = [map(lambda x: idx2labels[x], y) for y in test_y]
  words_test = [map(lambda x: idx2words[x], w) for w in test_x]

  res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')

  if res_test['f1'] > bestF1:
    rnn.save(folder)
    bestF1 = res_test['f1']
    print 'NEW BEST: epoch', e, 'best test F1', res_test['f1'], ' '*20
    s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
    s['be'] = e
    subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
  else:
    print ''

print 'BEST RESULT: epoch', e, 'best test F1', s['tf1'], 'with the model', folder
