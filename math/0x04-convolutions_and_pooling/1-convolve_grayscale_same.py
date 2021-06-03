#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph = int(kh / 2)
    pw = int(kw / 2)
    output = np.zeros((m, h, w))
    pad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=pad,
                     mode='constant', constant_values=0)
    for i in range(h):
        for j in range(w):
            image = imagesp[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(np.multiply(image, kernel),
                                     axis=(1, 2))
    return output
