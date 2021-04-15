#!/usr/bin/env python3
"""matrix_transpose function"""


def matrix_transpose(matrix):
    """
        Get the transpose of a 2D matrix: matrix to calculate
        Return: 2D transpose of matrix
    """
    transpose = []
    for i in range(len(matrix[0])):
        inner = []
        for j in range(len(matrix)):
            inner.append(matrix[j][i])
        transpose.append(inner)
    return transpose
