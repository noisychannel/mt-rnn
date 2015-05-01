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

  def __init__(self, nh, nc, ne, de):
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
    #idxs = T.imatrix() # columns = no of words in window, rows = len of sentence
    ## Prepare to recieve input and output labels
    #x = self.embeddings[idxs].reshape((idxs.shape[0], de*cs))
    X = T.imatrix('X')
    Y = T.imatrix('Y')
    Y_IDX = T.imatrix('Y_IDX')
    x = T.ivector('x')
    y = T.ivector('y')
    y_idx = T.ivector('y_idx')

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
        outputs_info=[self.h0],
        n_steps=x.shape[0])

    # Compute the context vector
    c = T.tanh(T.dot(self.V_e, h_e[-1]))

    ########## DECODER ###########

    # Initialize the hidden state
    h_d_0 = self.tanh(T.dot(self.V_d, c))

    #def decoder_recurrence(y_tm1, h_tm1, c):
    def decoder_recurrence(t, h_tm1, y_idx, c, y):
      # Get the previous y
      if t == 0:
        y_tm1 = y0
      else:
        y_tm1 = y[t-1]

      r = T.nnet.sigmoid(T.dot(self.W_r_d, y_tm1) + T.dot(self.U_r_d, h_tm1) + T.dot(self.C_r_d, c))
      # Update gate
      z = T.nnet.sigmoid(T.dot(self.W_z_d, y_tm1) + T.dot(self.U_z_d, h_tm1) + T.dot(self.C_z_d, c))
      # Gated output
      h_prime = T.tanh(T.dot(self.W_d, y_tm1) +  r * (T.dot(self.U_d, r * h_tm1) + T.dot(self.C_d, c)))
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
      # TODO:Only return NLL : Select NLL for the observed y
      nll_t = - T.log(p_t[y_idx[t], 0])
      return [h_t, nll_t]

    # TODO: Check output info
    # TODO: y_0 should be empty
    [h, nll], _ = theano.scan(fn=decoder_recurrence,
        sequences = T.arange(y.shape(0)),
        outputs_info = [h_d_0, None],
        non_sequences = [y_idx, c, y],
        n_steps = y.shape[0])

    # Average phrase negative log likelihood
    phrase_nll = T.mean(nll)

    self.phrase_train = theano.function(inputs=[x,y,y_idx], outputs=phrase_nll)

    # Get the phrase_nlls for all sentences by iterating over the batch
    phrase_nlls, _ = theano.scan(fn=self.phrase_train,
        sequences= [X, Y, Y_IDX],
        outputs_info = None,
        n_steps=X.shape[0])

    # Learning rate
    lr = T.scalar('lr')

    # Average batch negative log likelihood
    batch_nll = T.mean(phrase_nlls)

    # Compute paramter wise gradients
    batch_gradients = T.grad(batch_nll, self.params)
    # Compute updates wrt the batch
    batch_updates = OrderedDict((p, p - lr*g) for p,g in zip(self.params, batch_gradients))

    # Compile functions
    self.batch_train = theano.function(inputs=[X,Y,Y_IDX,lr], outputs=batch_nll, updates=batch_updates)
    self.batch_test = theano.function(inputs=[X,Y,Y_IDX], outputs=batch_nll)

  def save(self, folder):
    for param, name in zip(self.params, self.names):
      numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
