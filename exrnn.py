#!/usr/bin/env python

import os
import theano, numpy
from theano import tensor as T
from collections import OrderedDict

class RNNSLU(object):
  """ Elman neural net"""

  def __init__(self, nh, nc, ne, de, cs):
    """
    Hyperparameters used for initialization
    nh : dimension of the hidden layer
    nc : number of classes (labels)
    ne : size of vocabulary
    de : dimension of embedding
    cs : word context window size
    """
    # Parameter to be learnt : word embeddings
    self.embeddings = theano.shared(name='embeddings',
        value = 0.2 * numpy.random.uniform(-1.0, 1.0, (ne + 1, de))
        .astype(theano.config.floatX))

    # Parameter to be learnt : Weight matrix mapping input to the hidden layer (de*cs x nh)
    self.wx = theano.shared(name='wx',
        value = 0.2 * numpy.random.uniform(-1.0, 1.0, (de * cs, nh))
        .astype(theano.config.floatX))

    # Parameter to be learnt : Weight matrix mapping hidden layer from the
    # previous time step to the current one
    self.wh = theano.shared(name='wh',
        value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
        .astype(theano.config.floatX))

    # Parameter to be learnt : Weight matrix mapping hidden to output layer (nh x nc)
    self.w = theano.shared(name='w',
        value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nc))
        .astype(theano.config.floatX))

    # Parameter to be learnt : Bias at the hidden layer
    self.bh = theano.shared(name='bh',
        value = numpy.zeros(nh,
          dtype=theano.config.floatX))

    # Parameter to be learnt : The bias of the output layer
    self.b = theano.shared(name='b',
        value = numpy.zeros(nc,
          dtype=theano.config.floatX))

    # Parameter to be learnt : The hidden layer at time t=0
    self.h0 = theano.shared(name='h0',
        value = numpy.zeros(nh,
          dtype=theano.config.floatX))

    # Bundle the parameters
    self.params = [self.embeddings, self.wx, self.wh, self.w, self.bh, self.b, self.h0]
    self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']

    # Compile training function
    self.prepare_train(de, cs)

  def prepare_train(self, de, cs):
    """
    Trains the recurrent neural net
    """
    idxs = T.imatrix() # columns = no of words in window, rows = len of sentence
    # Prepare to recieve input and output labels
    x = self.embeddings[idxs].reshape((idxs.shape[0], de*cs))
    y = T.iscalar('y')

    def recurrence(x_t, h_tm1):
      """
      x_t : Input at time t
      h_tm1 : Hidden state at time t-1
      """
      # Compute the hidden state at time time
      # h_t = g(x_t . w_x + h_tm1 . w_h + b_h)
      h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
      # Compute the output layer
      # s_t = g(h_t . w + b)
      s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
      return [h_t, s_t]

    [h,s], _ = theano.scan(fn=recurrence,
        sequences=x,
        outputs_info=[self.h0, None],
        n_steps=x.shape[0])

    #print h.ndim
    #print s.ndim

    # TODO: What is the structure of s? What does the selection of axis do ?
    p_y_given_sentence = s[:,0,:]
    y_pred = T.argmax(p_y_given_sentence, axis=1)

    # Learning rate
    lr = T.scalar('lr')
    # Sentence negative log-likelihood (The objective function)
    sentence_nll = - T.mean(T.log(p_y_given_sentence)[T.arange(x.shape[0]), y])
    # Compute paramter wise gradients
    sentence_gradients = T.grad(sentence_nll, self.params)
    # Compute updats
    sentence_updates = OrderedDict((p, p - lr*g) for p,g in zip(self.params, sentence_gradients))

    # Compile functions
    self.classify = theano.function(inputs=[idxs], outputs=y_pred)
    self.sentence_train = theano.function(inputs=[idxs, y, lr], outputs=sentence_nll, updates=sentence_updates)

    # Normalize after each update : TODO: What is this update doing ?
    self.normalize = theano.function(inputs=[],
        updates={self.embeddings: self.embeddings / T.sqrt(self.embeddings ** 2).sum(axis=1).dimshuffle(0, 'x')})

  def save(self, folder):
    for param, name in zip(self.params, self.names):
      numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
