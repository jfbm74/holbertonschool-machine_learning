#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Class constructor
        nx is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # weights vector for the neuron
        # default mean is 0
        # default stddev is 1
        self.__W = np.random.randn(1, nx)
        # bias for the neuron
        self.__b = 0
        # activated output of the neuron (prediction)
        self.__A = 0

    # getter functions
    @property
    def W(self):
        """Retrieves the weights vector"""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias"""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output"""
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = (1 / (1 + np.exp(-z)))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        c = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                              (np.log(1.0000001 - A)))
        return c

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A = self.forward_prop(X)
        a = np.where(A < 0.5, 0, 1)
        return A, self.cost(Y, A)
