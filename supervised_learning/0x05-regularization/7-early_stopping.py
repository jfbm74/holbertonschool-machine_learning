#!/usr/bin/env python3
"""Function that creates a layer of a neural network using dropout"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that creates a layer of a neural network using dropout"""
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
