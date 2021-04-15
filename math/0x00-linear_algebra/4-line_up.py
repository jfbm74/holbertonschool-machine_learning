#!/usr/bin/env python3
""" adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    if len(arr1) != len(arr2):
        return None

    add_matrix = []
    for x, y in zip(arr1, arr2):
        add_matrix.append(x + y)
    return add_matrix
