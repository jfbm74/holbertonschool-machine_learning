#!/usr/bin/env python3
"""Module that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """test a neural network
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
