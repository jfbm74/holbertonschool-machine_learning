#!/usr/bin/env python3
"""module that builds a dense block as described in
Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that returns The concatenated output of
    each layer within the Dense Block and  the number of
    filters within the concatenated outputs, respectively"""

    for _ in range(layers):
        BN1 = K.layers.BatchNormalization(axis=3)(X)
        ReLU1 = K.layers.Activation('relu')(BN1)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer='he_normal')(ReLU1)
        BN2 = K.layers.BatchNormalization(axis=3)(conv1)
        ReLU2 = K.layers.Activation('relu')(BN2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer='he_normal')(ReLU2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters
