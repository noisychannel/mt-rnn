#!/usr/bin/env python

import os
import theano, numpy
import theano.typed_list
from theano import tensor as T
from collections import OrderedDict

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'
theano.traceback_limit = -1
#theano.profile = True
#theano.profile_optimizer = True
#theano.profile_memory = True

"""
Implements the RNN encoder-decoder framework from Cho et al.
"""

class RNNED(object):
  """ RNN encoder-decoder """

  def __init__(self, nh, nc, de):
    """
    Hyperparameters used for initialization
    nh : dimension of the hidden layer
    nc : number of classes (labels)
    de : dimension of embedding
    """

    # Parameter to be learnt : The hidden layer at time t=0
    self.h0 = theano.shared(name='h0',
        value = numpy.zeros(nh,
          dtype=theano.config.floatX))

    self.y0 = theano.shared(name='y0',
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

    # TODO: Changed from 2*de to de because the maxout unit does not work
    self.O_h = theano.shared(name='O_h',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, nh))
                .astype(theano.config.floatX))

    # TODO: Changed from 2*de to de because the maxout unit does not work
    self.O_y = theano.shared(name='O_y',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, de))
                .astype(theano.config.floatX))

    # TODO: Changed from 2*de to de because the maxout unit does not work
    self.O_c = theano.shared(name='O_c',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, de))
                .astype(theano.config.floatX))

    self.G = theano.shared(name='G',
                value=0.2 * numpy.random.uniform(-1.0, 1.0, (nc, de))
                .astype(theano.config.floatX))

    # Bundle the parameters
    self.encoderParams = [self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.V_e]
    self.encoderNames = ['h0', 'W_e', 'U_e', 'W_z_e', 'U_z_e', 'W_r_e', 'U_r_e', 'V_e']
    self.decoderParams = [self.V_d, self.W_d, self.U_d, self.C_d, self.W_z_d, self.U_z_d, self.C_z_d, self.W_r_d, \
        self.U_r_d, self.C_r_d, self.O_h, self.O_y, self.O_c, self.G]
    self.decoderNames = ['V_d', 'W_d', 'U_d', 'C_d', 'W_z_d', 'U_z_d', 'C_z_d', 'W_r_d', 'U_r_d', 'C_r_d', 'O_h', 'O_y', 'O_c', 'G']
    self.params = self.encoderParams + self.decoderParams
    self.names = self.encoderNames + self.decoderNames
    self.batch_gradients = OrderedDict((p,[]) for p in self.params)

    # Compile training function
    self.prepare_train(de)

  def prepare_train(self, de):
    """
    Trains the recurrent neural net
    """
    ## Prepare to recieve input and output labels
    X = T.fmatrix('X')
    Y = T.fmatrix('Y')
    Y_IDX = T.ivector('Y_IDX')

    ######### ENCODER ##########

    def encoder(x):
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
          outputs_info=self.h0,
          n_steps=x.shape[0])

      ## Compute the context vector
      c = T.tanh(T.dot(self.V_e, h_e[-1]))
      return c


    ########## DECODER ###########

    ## Average phrase negative log likelihood
    def decoder(x,y, y_idx):
      c = encoder(x)
      # Initialize the hidden state
      h_d_0 = T.tanh(T.dot(self.V_d, c))

      def decoder_recurrence(t, h_tm1, y_idx, c, y):
        # Get the previous y
        if t == 0:
          y_tm1 = self.y0
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
        # TODO: Changed from using maxout to regural combination
        #s_prime = T.dot(self.O_h, h_t) + T.dot(self.O_y, y_tm1) + T.dot(self.O_c, c)
        s = T.dot(self.O_h, h_t) + T.dot(self.O_y, y_tm1) + T.dot(self.O_c, c)
        # Maxout unit
        #s = numpy.empty((de))
        #s = numpy.zeros((de))
        #for i in range(de):
          ## TODO: This does not work right
          #s[i] = s_prime[2*i,0]
          #print s_prime.ndim
          #s[i] = T.max(s_prime[2*i:2*i+2], axis=0)[0]

        # Softmax to get probabilities over target vocab
        p_t = T.nnet.softmax(T.dot(self.G, s))
        # TODO:Only return NLL : Select NLL for the observed y
        #nll_t = - T.log(p_t[y_idx[t]])
        #theano.printing.debugprint(y_idx)
        nll_t = - T.log(p_t[0])
        return [h_t, nll_t]

      [h, nll], _ = theano.scan(fn=decoder_recurrence,
          sequences = T.arange(y.shape[0]),
          outputs_info = [h_d_0, None],
          non_sequences = [y_idx, c, y],
          n_steps = y.shape[0])

      phrase_nll = T.mean(nll)
      return phrase_nll, T.grad(phrase_nll, self.params)

    # Learning rate
    lr = T.scalar('lr')

    # Average batch negative log likelihood
    #theano.printing.debugprint(batch_nll)
    #theano.printing.pydotprint(batch_nll, "nll.png", compact=True, var_with_name_simple=True)
    # Average phrase negative log likelihood
    phrase_nll, phrase_gradients = decoder(X,Y,Y_IDX)
    phrase_updates = OrderedDict((p, p - lr*g) for p,g in zip(self.params, phrase_gradients))
    # Accumulate gradients so that they can be averaged and applied later
    #for p,g in zip(self.params, phrase_gradients):
      #self.batch_gradients[p].append(g)

    self.phrase_train = theano.function(inputs=[X,Y,Y_IDX,lr], outputs=phrase_nll, on_unused_input='warn', updates=phrase_updates)
    self.phrase_test = theano.function(inputs=[X,Y,Y_IDX], outputs=phrase_nll, on_unused_input='warn')

  def train(self, batch, lr):
    # Accumulates gradients for the batch so that they can be averaged and applied
    #self.batch_gradients = OrderedDict((p,[]) for p in self.params)
    #print self.W_d.get_value()
    #batch_size = len(batch)
    #phrase_nll = 0
    for (x,y,y_idx) in batch:
      self.phrase_train(x,y,y_idx,lr)

    #print self.batch_gradients
    #for p, g in self.batch_gradients.iteritems():
      #p = p - lr * T.mean(numpy.asarray(g))

    #print float(phrase_nll) / batch_size

  def test(self, batch):
    batch_size = len(batch)
    batch_nll = 0
    for (x,y,y_idx) in batch:
      batch_nll += self.phrase_test(x,y,y_idx)

    print "Test average LL = ", float(batch_nll)/batch_size


  def save(self, folder):
    for param, name in zip(self.params, self.names):
      numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
