#!/usr/bin/env python3
"""Function that calculates the accuracy of a prediction"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    y_post = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    equal = tf.equal(y_post, y_pred)
    equal = tf.cast(equal, tf.float32)
    accuracy = tf.reduce_mean(equal)
    return accuracy
