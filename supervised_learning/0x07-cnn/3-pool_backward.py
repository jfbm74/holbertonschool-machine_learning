#!/usr/bin/env python3
"""
Module Performs back propagation over a pooling layer of a neural network
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs back propagation over a pooling layer of a
     neural network
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    s_h = h * sh
                    s_w = w * sw
                    box = dA[i, h, w, c]
                    if mode == 'max':
                        tmp = A_prev[i, s_h:kh+s_h, s_w:kw+s_w, c]
                        mask = (tmp == np.max(tmp))
                        dA_prev[i, s_h:kh+s_h, s_w:kw+s_w, c] += box * mask
                    if mode == 'avg':
                        dA_prev[i, s_h:kh+s_h, s_w:kw+s_w, c] += box/kh/kw
    return dA_prev
