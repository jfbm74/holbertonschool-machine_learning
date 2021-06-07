#!/usr/bin/env python3
""" Module defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Function for deep neural network performing binary classification """

    @staticmethod
    def Weights_init(nx, layers):
        """ Weights initialization"""
        weights = {}
        for l in range(len(layers)):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError('layers must be a list of positive integers')
            ant_layer = layers[l - 1] if l > 0 else nx
            weights.update({
                'W' + str(l + 1): np.random.randn(
                    layers[l], ant_layer) * np.sqrt(2 / ant_layer),
                'b' + str(l + 1): np.zeros((layers[l], 1))})
        return weights

    def __init__(self, nx, layers):
        """ Neural Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = self.Weights_init(nx, layers)
