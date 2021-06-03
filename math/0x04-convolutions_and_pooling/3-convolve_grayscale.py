#!/usr/bin/env python3
""" This module contains the function convolve_grayscale. """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ This module contains the function convolve_grayscale. """
    kh, kw = kernel.shape
    m, imh, imw = images.shape
    sh, sw = stride
    if type(padding) == tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int(((imh - 1) * sh - imh + kh) / 2) + 1
        pw = int(((imw - 1) * sw - imw + kw) / 2) + 1
    else:
        ph = pw = 0
    pad = np.pad(images, ((0,), (ph,), (pw,)))
    varh = int((imh + 2 * ph - kh) / sh + 1)
    varw = int((imw + 2 * pw - kw) / sw + 1)
    output = np.zeros((m, varh, varw))
    for i in range(varh):
        for j in range(varw):
            x = i * sh
            y = j * sw
            output[:, i, j] = (pad[:, x: x + kh, y: y + kw] *
                               kernel).sum(axis=(1, 2))
    return output
