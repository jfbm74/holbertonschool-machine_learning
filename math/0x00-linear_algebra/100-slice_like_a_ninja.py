#!/usr/bin/env python3
""" slices a matrix along specific axes:"""


def np_slice(matrix, axes={}):
    """ slices a matrix along specific axes:"""
    new = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        new[key] = slice(*value)
    return matrix[tuple(new)]
