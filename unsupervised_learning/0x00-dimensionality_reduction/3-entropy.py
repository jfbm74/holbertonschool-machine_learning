#!/usr/bin/env python3
"""Calculate Shannon entropies and P affinities relative to a data point."""


import numpy as np


def HP(Di, beta):
    """
    Returns: (Hi, Pi)
    """
    Pi = np.exp(-Di * beta)
    sumPi = np.sum(Pi)
    Pi = Pi / sumPi
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
