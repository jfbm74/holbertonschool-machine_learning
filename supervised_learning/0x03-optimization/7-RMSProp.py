#!/usr/bin/env python3
"""Function that update a variable using
the RMSProp"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Function that update a variable
    using the RMSProp """
    Sd = np.multiply(beta2, s) + np.multiply((1 - beta2), grad ** 2)
    var = var - np.multiply(alpha, np.divide(grad, ((Sd ** 0.5) + epsilon)))
    return var, Sd
