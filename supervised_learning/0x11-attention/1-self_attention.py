#!/usr/bin/env python3
""" RNN's """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class Self attention
    """
    def __init__(self, units):
        """
        init function
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        call function
        """
        s_prev1 = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(s_prev1) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        contx = tf.reduce_sum(weights * hidden_states, axis=1)

        return contx, weights
