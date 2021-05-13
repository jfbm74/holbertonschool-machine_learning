#!/usr/bin/env python3
"""
Returns two placeholders, x and y, for the neural network
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
