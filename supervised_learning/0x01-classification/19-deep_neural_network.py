#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """Class a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                     self.nx) *
                                                     np.sqrt(2/self.nx))
            else:
                self.__weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                     layers[i-1]) *
                                                     np.sqrt(2/layers[i-1]))
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Retrieves L"""
        return self.__L

    @property
    def cache(self):
        """Retrieves cache"""
        return self.__cache

    @property
    def weights(self):
        """Retrieves weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            Z = (np.matmul(self.__weights["W{}".format(i + 1)],
                           self.__cache["A{}".format(i)]) +
                 self.__weights["b{}".format(i + 1)])
            self.__cache["A{}".format(i + 1)] = (np.exp(Z) / (np.exp(Z) + 1))
        return self.__cache["A{}".format(i + 1)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        return ((-1/m)*np.sum(np.multiply(Y, np.log(A)) +
                             np.multiply((1-Y), np.log(1.0000001 - A))))
