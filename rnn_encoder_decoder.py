#!/usr/bin/env python

import os
import theano, numpy
from theano import tensor as T
from collections import OrderedDict

"""
Implements the RNN encoder-decoder framework from Cho et al.
"""

class RNNED(object):
  """ RNN encoder-decoder """

  def __init__(self, nh, nc, ne, de, cs):
    """
    Hyperparameters used for initialization
    nh : dimension of the hidden layer
    nc : number of classes (labels)
    ne : size of vocabulary
    de : dimension of embedding
    cs : word context window size
    """
    # Parameter to be learnt : The hidden layer at time t=0
    self.h0 = theano.shared(name='h0',
        value = numpy.zeros(nh,
          dtype=theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state
    self.W_e = theano.shared(name='W_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state
    self.U_e = theano.shared(name='U_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state : Reset gate
    self.W_z_e = theano.shared(name='W_z_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state : Reset gate
    self.U_z_e = theano.shared(name='U_z_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state : Update gate
    self.W_r_e = theano.shared(name='W_r_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the encoder hidden state : Update gate
    # TODO: Add bias paramters
    self.U_r_e = theano.shared(name='U_r_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the computation of the context vector
    self.V_e = theano.shared(name='V_e',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (de, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the initialization of the hidden state of the decoder
    self.V_d = theano.shared(name='V_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state
    self.W_d = theano.shared(name='W_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state
    self.U_d = theano.shared(name='U_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state
    self.C_d = theano.shared(name='C_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Update gate
    self.W_z_d = theano.shared(name='W_z_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Update gate
    self.U_z_d = theano.shared(name='U_z_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Update gate
    self.C_z_d = theano.shared(name='C_z_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Reset gate
    self.W_r_d = theano.shared(name='W_r_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Reset gate
    # TODO: Add bias paramters
    self.U_r_d = theano.shared(name='U_r_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh))
                .astype(theano.config.floatX))

    # Parameter : Weight matrix for the decoder hidden state : Reset gate
    self.C_r_d = theano.shared(name='C_r_d',
                value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de))
                .astype(theano.config.floatX))

    self.O_h = theano.shared(name='O_h',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (2*de, nh))
                .astype(theano.config.floatX))

    self.O_y = theano.shared(name='O_y',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (2*de, de))
                .astype(theano.config.floatX))

    self.O_c = theano.shared(name='O_c',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (2*de, de))
                .astype(theano.config.floatX))

    self.G = theano.shared(name='G',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (nc, de))
                .astype(theano.config.floatX))

    # Parameter to be learnt : word embeddings
    self.embeddings = theano.shared(name='embeddings',
        value = 0.2 * numpy.random.uniform(-1.0, 1.0, (ne + 1, de))
        .astype(theano.config.floatX))

    # Bundle the parameters
    self.encoderParams = [self.h0, self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.V_e]
    self.encoderNames = ['h0', 'W_e', 'U_e', 'W_z_e', 'U_z_e', 'W_r_e', 'U_r_e', 'V_e']
    self.decoderParams = [self.V_d, self.W_d, self.U_d, self.C_d, self.W_z_d, self.U_z_d, self.C_z_d, self.W_r_d, \
        self.U_r_d, self.C_r_d, self.O_h, self.O_y, self.O_c, self.G]
    self.decoderNames = ['V_d', 'W_d', 'U_d', 'C_d', 'W_z_d', 'U_z_d', 'C_z_d', 'W_r_d', 'U_r_d', 'C_r_d', 'O_h', 'O_y', 'O_c', 'G']
    self.params = self.encoderParams + self.decoderParams
    self.names = self.encoderNames + self.decoderNames

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

    ######### ENCODER ##########

    def encoder_recurrence(x_t, h_tm1):
      # Reset gate
      r = T.nnet.sigmoid(T.dot(self.W_r_e, x_t) + T.dot(self.U_r_e, h_tm1))
      # Update gate
      z = T.nnet.sigmoid(T.dot(self.W_z_e, x_t) + T.dot(self.U_z_e, h_tm1))
      # Gated output
      h_prime = T.tanh(T.dot(self.W_e, x_t) + T.dot(self.U_e, r * h_tm1))
      # Compute hidden state
      h_t = z * h_tm1 + (1 - z) * h_prime
      return h_t

    # Encoder Recurrence
    h_e, _ = theano.scan(fn=encoder_recurrence,
        sequences = x,
        outputs_info=[self.h0, None],
        n_steps=x.shape[0])

    # Compute the context vector
    c = T.tanh(T.dot(self.V_e, h_e[-1]))

    ########## DECODER ###########

    # Initialize the hidden state
    h_d_0 = self.tanh(T.dot(self.V_d, c))

    def decoder_recurrence(y_tm1, h_tm1, c):
      r = T.nnet.sigmoid(T.dot(self.W_r_d, y_tm1) + T.dot(self.U_r_d, h_tm1) + T.dot(self.C_r_d, c))
      # Update gate
      z = T.nnet.sigmoid(T.dot(self.W_z_d, y_tm1) + T.dot(self.U_z_d, h_tm1) + T.dot(self.C_z_d, c))
      # Gated output
      h_prime = T.tanh(T.dot(self.W_d, y_tm1) + T.dot(self.U_d, r * h_tm1) + T.dot(self.C_d, c))
      # Compute hidden state
      h_t = z * h_tm1 + (1 - z) * h_prime
      # Compute the activation based on y_tm1, c and h_t
      s_prime = T.dot(self.O_h, h_t) + T.dot(self.O_y, y_tm1) + T.dot(self.O_c, c)
      # Maxout unit
      s = numpy.empty([self.de])
      for i in range(self.de):
        s[i] = max(s_prime[2*i], s_prime[2*i + 1])

      # Softmax to get probabilities over target vocab
      p_t = T.nnet.softmax(T.dot(self.G, s))
      return h_t, p_t

    # TODO: Check output info
    # TODO: y_0 should be empty
    [h,p], _ = theano.scan(fn=decoder_recurrence,
        sequences=y,
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
