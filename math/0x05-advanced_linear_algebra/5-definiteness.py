#!/usr/bin/env python3
"""
Calculates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """

    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None

    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    w, _ = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
