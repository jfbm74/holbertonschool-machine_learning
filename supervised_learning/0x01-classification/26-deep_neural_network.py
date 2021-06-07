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
                self.__weights["W{}".format(i + 1)] = \
                    (np.random.randn(layers[i],
                                     self.nx) *
                     np.sqrt(2 / self.nx))
            else:
                self.__weights["W{}".format(i + 1)] = \
                    (np.random.randn(layers[i],
                                     layers[i - 1]) *
                     np.sqrt(2 / layers[i - 1]))
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

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
        return ((-1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply((1 - Y), np.log(1.0000001 - A))))

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        pred, _ = self.forward_prop(X)
        return np.where(pred < 0.5, 0, 1), self.cost(Y, pred)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent on the neural network"""
        m = Y.shape[1]
        dz_prev = []
        weights = self.__weights.copy()
        for n in range(self.__L, 0, -1):
            A = cache.get('A' + str(n))
            A_prev = cache.get('A' + str(n - 1))
            wx = weights.get('W' + str(n + 1))
            bx = weights.get('b' + str(n))
            if n == self.__L:
                dz = A - Y
            else:
                dz = np.matmul(wx.T, dz_prev) * (A * (1 - A))
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz_prev = dz
            w = weights.get('W' + str(n))
            self.__weights.update({
                'W' + str(n): w - (dw * alpha),
                'b' + str(n): bx - (db * alpha)})

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Method that Trains the deep neural network """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        costs = {}
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if verbose is True and i % step == 0:
                print('Cost after ' + str(i) + ' iterations: '
                      + str(self.cost(Y, A)))
                costs[i] = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)
        cost_list = costs.items()
        cost_list = sorted(cost_list)
        x, y = zip(*cost_list)
        plt.plot(x, y)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Training Cost')
        plt.show()
        return self.evaluate(X, Y)
