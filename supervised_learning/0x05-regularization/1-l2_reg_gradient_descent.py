#!/usr/bin/env python3
"""Function that updates the weights and biases of a neural network using
 gradient descent with L2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """A function that updates the weights and biases of a neural network using
     gradient descent with L2 regularization"""
    derz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        L2 = (lambtha / Y.shape[1]) * weights["W" + str(i)]
        derb = (np.sum(derz, axis=1, keepdims=True) / Y.shape[1])
        derw = (np.matmul(derz, cache["A" + str(i - 1)].T) / Y.shape[1]) + L2
        derz = np.matmul(weights["W" + str(
            i)].T, derz) * (1 - (cache["A" + str(i - 1)] ** 2))
        weights["b" + str(i)] = weights["b" + str(
            i)] - (alpha * derb)
        weights["W" + str(i)] = weights["W" + str(i)] - (alpha * derw)
