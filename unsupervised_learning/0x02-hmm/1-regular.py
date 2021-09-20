#!/usr/bin/env python3
"""Steady state probabilities of a regular markov chain"""

import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of a regular markov
    """
    try:
        if len(P.shape) != 2:
            return None
        n = P.shape[0]
        if n != P.shape[1]:
            return None

        evals, evecs = np.linalg.eig(P.T)

        state = (evecs / evecs.sum())

        new_state = np.dot(state.T, P)
        for i in new_state:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
