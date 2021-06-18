#!/usr/bin/env python3
"""Module that builds a transition layer as described
in Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that returns the output of the transition
    layer and the number of filters within the output"""
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    length = int(nb_filters * compression)
    X = K.layers.Conv2D(length, kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.AvgPool2D((2, 2), padding='same')(X)
    return X, length
