#!/usr/bin/env python3
""" Calculates the F1 score of a confusion matrix """


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Calculates the F1 score of a confusion matrix """
    score = 2 * precision(confusion) * sensitivity(confusion) / \
        (precision(confusion) + sensitivity(confusion))

    return score
