#!/usr/bin/env python3
"""
Creates a layer
"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that returns the tensor output of the layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            name="layer",
                            kernel_initializer=w)
    return layer(prev)
