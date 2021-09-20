#!/usr/bin/env python3
"""Markov chain probability"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that deteermines the probability of a markov chain
    """
    try:

        if len(P.shape) != 2:
            return None
        n1, n2 = P.shape
        if (n1 != n2) or type(P) is not np.ndarray or not isinstance(t, int):
            return None
        if t < 0:
            return None
        if n1 != s.shape[1] or s.shape[0] != 1:
            return None
        for i in range(t):
            # formula: p(t)ij = ∑^rk=1 p(ik)p(kj) .
            # dot pro product between the sate and the P given n iterations
            s = np.dot(s, P)
        return s
    except Exception:
        return None
