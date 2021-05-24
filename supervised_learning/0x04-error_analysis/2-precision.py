#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""


import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
