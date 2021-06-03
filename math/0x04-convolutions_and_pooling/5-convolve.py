#!/usr/bin/env python3
""" Performs a convolution on images using multiple kernels """

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""

    kh, kw, _, nc = kernels.shape
    m, imh, imw, c = images.shape
    sh, sw = stride
    if type(padding) == tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int(((imh - 1) * sh - imh + kh) / 2) + 1
        pw = int(((imw - 1) * sw - imw + kw) / 2) + 1
    else:
        ph = pw = 0
    padded = np.pad(images, ((0,), (ph,), (pw,), (0,)))
    varh = int((imh + 2 * ph - kh) / sh + 1)
    varw = int((imw + 2 * pw - kw) / sw + 1)
    output = np.zeros((m, varh, varw, nc))
    for i in range(varh):
        for j in range(varw):
            for k in range(nc):
                x = i * sh
                y = j * sw
                output[:, i, j, k] = (padded[:, x: x + kh, y: y + kw, :] *
                                      kernels[:, :, :, k]).sum(axis=(1, 2, 3))
    return output
