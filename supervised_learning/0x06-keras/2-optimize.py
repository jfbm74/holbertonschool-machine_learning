#!/usr/bin/env python3
"""Module that sets up Adam optimization for a keras model """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    function def optimize_model(network, alpha, beta1, beta2):
    that sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics:
    """
    optimize_adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=optimize_adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
