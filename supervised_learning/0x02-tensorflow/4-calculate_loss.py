#!/usr/bin/env python3
"""
Calculates the softmax cross-entropy loss of pred
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of pred"""
    tf.losses.softmax_cross_entropy(y, y_pred)
    return tf.losses.softmax_cross_entropy(y, y_pred)
