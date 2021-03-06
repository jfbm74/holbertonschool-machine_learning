#!/usr/bin/env python3
""" RNN encoder"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    class rnn encoder
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Init function
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        init class hidden
        """
        initializer = tf.keras.initializers.Zeros()
        # Q equals to matrix
        hiddenQ = initializer(shape=(self.batch, self.units))
        return hiddenQ

    def call(self, x, initial):
        """
        Call methods
        """
        embedding = self.embedding(x)
        outputs, last_hiddenQ = self.gru(embedding,
                                         initial_state=initial)
        return outputs, last_hiddenQ
