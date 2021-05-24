#!/usr/bin/env python3
""" Calculates the sensitivity for each class in a confusion matrix """


import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity for each class in a confusion matrix """
    return np.diagonal(confusion) / np.sum(confusion, axis=1)
