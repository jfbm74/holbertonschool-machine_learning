#!/usr/bin/env python3
"""Calculate intersection of data given hypothetical probabilities"""


import numpy as np
import scipy.special as special


def intersection(x, n, P, Pr):
    """ calculates  intersection of obtaining this data
            with the various hypothetical probabilities
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater " +
                         "than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not (np.all(P >= 0) and np.all(P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not (np.all(Pr >= 0) and np.all(Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(1, np.sum(Pr)):
        raise ValueError("Pr must sum to 1")

    def factorial(m):
        """ calculates factorial of n """
        return np.math.factorial(m)

    likelihood = np.ndarray(P.shape)
    pos = ((x / n))

    for p in range(len(P)):
        fact_n = factorial(n)
        likelihood[p] = ((factorial(n) /
                         (factorial(x) * factorial(n - x))) *
                         (np.power(P[p], x)) *
                         (np.power((1 - P[p]), (n - x))))
    return likelihood * Pr
