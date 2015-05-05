# RNN-encoder decoder for machine translation

Introduction
============

In phrase based machine translation, phrase pairs are typically
extracted using unsupervised alignment methods. These alignment methods
which are typically generative in nature are unable to incorporate
information about linguistic integrity and other measures of quality of
phrase pairs. Hence, the extracted phrase pairs are often fairly noisy.
One method of using these phrase pairs without deviating too far from
the conventional phrase extraction procedures is to use additional
features for each phrase pair and then learn the weights for these
features using *discriminative training* where the goal is to
discriminate between good and bad hypotheses. With the recent use of
*neural networks in machine translation*, we have the capability to
represent variable length sentences into a fixed size vector
representation. This representation can be created based on any measure
of quality we deem to be useful. Once we obtain vector representations
of sentences/phrases based on some property of the language (syntax,
semantics), it is relatively easy to ask ourselves how good a phrase
pair is. This project builds on this work and other work in neural
machine translation to estimate the *phrasal similarity* of phrase
pairs. Evaluation will be conducted by using this metric as another
feature in phrase based translation and for phrase table pruning.

Phrase Similarity
=================

A natural question to ask when using unsupervised alignment for phrase
extraction is : How do we trust the quality of the phrase pairs that we
have extracted ? Obviously, this questions assumes the presence of a
goodness function for phrase quality. One standard metric is to ask if
the source and the target components of a phrase pair convey the same
information (semantics). There are several linguistically motivated
methods to do this. We will describe a method that encodes semantic
information into a fixed sized vector using an unsupervised method and
then use it to measure phrase pair similarity. Before we proceed
further, let us look at some of the kinds of problems we may encounter
in an actual phrase table with phrase pairs.

-   **Rare phrases**: Rare phrase pair occurrences provide a sub-optimal
    estimate for phrase translation probabilities.\
    $p(\text{sorona} \;\;|\;\; \text{tristifical}) = 1$\
    $p(\text{tristifical} \;\;|\;\; \text{sorona}) = 1$

-   **Independence assumptions** : The choice to use one phrase pair
    over an another is largely independent of previous decisions.

-   **Segmentation** : Phrase segmentation is generally not
    linguistically motivated and a large percentage of the phrase pairs
    are not good translations.\
    (, veinte délares, era ||| you! twenty dollars, it was)\
    (Exactamente como ||| how they want to)

The RNN encoder-decoder framework
=================================

This project draws motivation from @schwenk_continuous_2012 which
demonstrates the use of a feed forward neural network in phrase based
machine translation. Specifically, it implements the RNN encoder-decoder
framework for estimating phrase similarity described in
@cho_learning_2014. It is worth noting that a similar exposition appears
in @sutskever_sequence_2014, while @bahdanau_neural_2014 extends this
framework to perform joint alignment and translation.

The RNN encoder-decoder framework consists of a pair of recurrent neural
networks (RNNs). The goal of the encoder is to *encode* a variable
length input sentence to a fixed size vector representation. The decoder
on the other hand, takes a fixed size vector representation and produces
a variable length sentence. In a probabilistic framework, this is akin
to learning the conditional distribution
$p(y_1, \cdots , y_{T'} | x_1, \cdots , x_{T})$ where $x$ and $y$ are
tokens from some language pair and $T$ is not necessarily equal to $T'$.

Encoder
-------

The encoder is a straightforward RNN which consists of a hidden state
$h$ which is updated at every time step $t$ while operating on some
sequence $x = (x_i,\cdots,x_T)$. That is
$$h_{t} = f(h_{t-1}, x_t) \nonumber$$ where $f$ is some non-linear
activation function. We will use a special form of the LSTMs
(@hochreiter_long_1997) as our activation function as described in
@cho_learning_2014 unless otherwise stated. The hidden state obtained
for the entire input sequence will be called the *context vector* $c$.

Decoder
-------

The decoder is a modified RNN that will produce the next symbol $y_t$
given the current hidden state $h_t$, previous symbol generated
$y_{t-1}$ and the context vector from the encoder. $c$ The hidden state
in turn is created as a function of the previous symbol generated
$y_{t-1}$, the context vector from the encoder $c$ and the previous
hidden state $h_{t-1}$. That is, $$\begin{aligned}
  & h_{t} = f(h_{t-1}, y_{t-1}, c) \\
  & p(y_t | y_{t-1}, \cdots , y_1, c) = g(h_t, y_{t-1}, c)\end{aligned}$$
where $f$ is an LSTM-like non-linear activation function and $g$ is a
logistic sigmoid (softmax) function (which is required to produce valid
probabilities).

Evaluation
==========

We will explore the validity of the task by two techniques on the
translation quality of a parallel dataset (to be decided):

1.  **Phrase features** : In a phrase based model, we plan to use this
    model to estimate phrasal similarity and use it as an additional
    feature in the phrase table. The SMT system will then use this
    feature for decoding during tuning and at test time. We hope to
    achieve better translation quality with this feature.

2.  **Phrase table pruning** : As an additional test of the usefulness
    of this feature, we will use it to prune an existing phrase table.
    The main idea is that the phrase table generally contains a large
    number of phrase pairs that are bad (given our goodness function),
    and any phrase pair below a certain threshold can be eliminated.
    This evaluation method will remain an optional investigation and
    will be performed if time permits.

Software
========

We plan to use two major software components for this project :

1.  **Theano** : A python based library that allows relatively easy
    implementation of GPU/CPU based operations for implementing neural
    networks.

2.  **Moses** : The phrase based translation system in Moses will be
    used and we plan to add an additional feature function for decoding.
    If found useful, we will contribute this feature to the Moses code
    base.

Interim Report
==============

This section reports updates with this project since the proposal was
submitted. There are two major components that have been finalized

1.  **Dataset and Baseline** : The French-English translation task from
    WMT-2013 will be used to evaluate this project. The dataset from the
    corresponding task and partitions corresponding to train, dev and
    test will be used to build the phrase table, tune and evaluate
    respectively. The baseline system will be a phrase based SMT system
    built using Moses.

2.  **Preliminary RNN implementation** : A prototype of the RNN that
    will be used in this task has been built and exists at
    <https://github.com/noisychannel/mt-rnn>. This version uses Theano
    and is built so that it can be adapted for any task. At the current
    moment, an example which allows one to predict tags with the ATIS
    data and evaluate using the CONLL evaluation setup is included.
