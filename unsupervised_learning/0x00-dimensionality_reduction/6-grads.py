#!/usr/bin/env python3
"""grads"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculate the gradients of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))

    PQ = P - Q
    PQ_expanded = np.expand_dims((PQ * num).T, axis=2)
    for i in range(n):
        y_diff = Y[i, :] - Y
        dY[i, :] = np.sum((PQ_expanded[i, :] * y_diff), 0)
    return dY, Q
