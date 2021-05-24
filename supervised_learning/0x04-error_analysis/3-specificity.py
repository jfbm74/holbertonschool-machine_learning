#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""


import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - TP - FP - FN
    specificity = TN / (FP + TN)
    return specificity
