#!/usr/bin/env python3
"""Performs a convolution on grayscale images custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images custom padding"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding[0], padding[1]
    nw = int(w - kw + (2 * pw) + 1)
    nh = int(h - kh + (2 * ph) + 1)
    output = np.zeros((m, nh, nw))
    pad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        for j in range(nw):
            image = imagesp[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(np.multiply(image, kernel),
                                     axis=(1, 2))
    return output
