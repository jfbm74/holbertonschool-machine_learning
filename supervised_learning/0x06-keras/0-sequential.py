#!/usr/bin/env python3
"""Module that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    function def build_model(nx, layers, activations, lambtha, keep_prob):
    that builds a neural network with the Keras library
    """
    l2reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        model.add(K.layers.Dense(
            layer, input_shape=(nx,),
            activation=activation,
            kernel_regularizer=l2reg))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))
    return model
