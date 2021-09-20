#!/usr/bin/env python3
"""containing the P affinities"""


import numpy as np


def cost(P, Q):
    """P is a numpy.ndarray of shape (n, n) containing the P affinities
    Returns: C, the cost of the transformation"""
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
