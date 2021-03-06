#!/usr/bin/env python3
"""Function that shuffles the data points in two matrices """

import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two matrices """
    random = np.random.permutation(X.shape[0])
    return X[random], Y[random]
