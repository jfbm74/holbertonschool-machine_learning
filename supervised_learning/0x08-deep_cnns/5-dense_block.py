#!/usr/bin/env python3
"""module that builds a dense block as described in
Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that returns The concatenated output of
    each layer within the Dense Block and  the number of
    filters within the concatenated outputs, respectively"""
    concat = X
    for i in range(layers):
        X = K.layers.BatchNormalization(axis=3)(concat)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate * 4, kernel_size=(1, 1),
                            padding='same',
                            strides=(1, 1),
                            kernel_initializer='he_normal')(X)
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, kernel_size=(3, 3), padding='same',
                            strides=(1, 1),
                            kernel_initializer='he_normal')(X)
        concat = K.layers.concatenate([concat, X], axis=3)
        filters = filters + growth_rate
    return concat, filters
