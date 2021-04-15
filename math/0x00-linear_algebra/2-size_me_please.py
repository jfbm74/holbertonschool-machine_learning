#!/usr/bin/env python3
"""
function matrix_shape(matrix): calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """Calculates the shape of a given matrix"""
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
