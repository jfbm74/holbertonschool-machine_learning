#!/usr/bin/env python3
""" This module contains the function pool. """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.
    """
    kh, kw = kernel_shape
    m, imh, imw, c = images.shape
    sh, sw = stride
    if mode == 'max':
        pool = np.max
    else:
        pool = np.average
    varh = int((imh - kh) / sh + 1)
    varw = int((imw - kw) / sw + 1)
    output = np.zeros((m, varh, varw, c))
    for i in range(varh):
        for j in range(varw):
            x = i * sh
            y = j * sw
            output[:, i, j, :] = pool(images[:, x: x + kh, y: y + kw, :],
                                      axis=(1, 2))
    return output
