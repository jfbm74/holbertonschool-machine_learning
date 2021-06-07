#!/usr/bin/env python3
""" Module that converts a numeric label vector into a
one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Function that converts a numeric label vector into
    a one-hot matrix
    """
    if type(classes) is not int or classes <= 0:
        return None
    if type(Y) is not np.ndarray:
        return None

    m = Y.shape[0]
    try:
        one_hot = np.zeros((classes, m))
        for i in range(m):
            row = Y[i]
            one_hot[row][i] = 1
        return one_hot
    except Exception:
        return None
