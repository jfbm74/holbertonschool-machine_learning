#!/usr/bin/env python3
"""Module that performs forward propagation over a pooling layer
 of a neural network"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation over a pooling layer
     of a neural network"""
    nw = int(((A_prev.shape[2] - kernel_shape[1]) / stride[1]) + 1)
    nh = int(((A_prev.shape[1] - kernel_shape[0]) / stride[0]) + 1)
    output = np.zeros((A_prev.shape[0], nh, nw, A_prev.shape[3]))
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            A_min = A_prev[:, x:x + kernel_shape[0],
                           y:y + kernel_shape[1], :]
            if mode == 'max':
                output[:, i, j, :] = np.max(A_min, axis=(1, 2))
            else:
                output[:, i, j, :] = np.average(A_min, axis=(1, 2))
    return output
