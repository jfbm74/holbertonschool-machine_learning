#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of a neural network using dropout"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(1 - keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=w,
                            kernel_regularizer=dropout,
                            name='layer')
    return layer(prev)
