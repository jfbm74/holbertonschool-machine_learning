#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt

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
        a = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        evaluate = a, cost
        return evaluate

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dw = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m
        # updating
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # The following unused i is not a problem
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)