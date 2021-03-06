#!/usr/bin/env python3
"""
Calculates the Q affinities
"""


import numpy as np


def Q_affinities(Y):
    """
    Returns: Q, num
    """
    sumY = np.sum(np.square(Y), 1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sumY).T, sumY))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / np.sum(num)
    return (Q, num)
