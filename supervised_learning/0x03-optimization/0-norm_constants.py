#!/usr/bin/env python3
"""Function that calculates the normalization """

import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization """
    mean = sum(X) / X.shape[0]
    X = X - mean
    desv = (sum(X ** 2) / X.shape[0]) ** 0.5
    return mean, desv
