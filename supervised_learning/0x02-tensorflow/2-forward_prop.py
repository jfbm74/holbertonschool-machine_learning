#!/usr/bin/env python3
""" Creates forward propagation graph """


import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Function that creates forward propagation graph """
    a = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        a = create_layer(a, layer_sizes[i], activations[i])
    return a
