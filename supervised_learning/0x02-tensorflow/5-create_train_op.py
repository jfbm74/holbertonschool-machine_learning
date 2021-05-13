#!/usr/bin/env python3
"""Function that creates the training operation for  network"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for  network"""
    optimus = tf.train.GradientDescentOptimizer(alpha)
    return optimus.minimize(loss)
