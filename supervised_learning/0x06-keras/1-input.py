#!/usr/bin/env python3
"""Module that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    function def build_model(nx, layers, activations, lambtha, keep_prob):
    that builds a neural network with the Keras library
    """
    income = K.Input(shape=(nx,))
    l2reg = K.regularizers.l2(lambtha)
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=layer,
            activation=activation,
            kernel_regularizer=l2reg
        )(x if i else income)
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)
    model = K.Model(inputs=income, outputs=x)
    return model
