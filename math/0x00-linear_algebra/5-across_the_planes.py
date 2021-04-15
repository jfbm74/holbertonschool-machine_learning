#!/usr/bin/env python3
"""Adds two matrices in 2D"""


def add_matrices2D(mat1, mat2):
    """ Add two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    add_mt = []
    for row1, row2 in zip(mat1, mat2):
        add_mt.append([])
        for x, y in zip(row1, row2):
            add_mt[-1].append(x + y)
    return add_mt
