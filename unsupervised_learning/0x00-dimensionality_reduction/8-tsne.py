#!/usr/bin/env python3
"""placeholder"""


import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    T-sne function
    """
    X = pca(X, idims)
    n, d = X.shape
    P = P_affinities(X, perplexity=perplexity) * 4
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    for i in range(iterations):
        dY, Q = grads(Y, P)

        if i <= 20:
            momentum = 0.5
        else:
            momentum = 0.8

        iY = momentum * iY - lr * dY
        Y = Y + iY - np.tile(np.mean(Y, 0), (n, 1))

        if (i + 1) != 0 and (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        if (i + 1) == 100:
            P = P / 4.

    return Y


