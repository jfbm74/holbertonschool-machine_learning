#!/usr/bin/env python3
""" defines function that concatenates two matrices along a specific axis """


def matrix_shape(matrix):
    """ size of matrix """
    matr = []
    while type(matrix) is list:
        matr.append(len(matrix))
        matrix = matrix[0]
    return matr


def cat_matrices(mat1, mat2, axis=0):
    """ contats 2 matrix """
    from copy import deepcopy
    temp_matrix1 = matrix_shape(mat1)
    temp_matrix2 = matrix_shape(mat2)
    if len(temp_matrix1) != len(temp_matrix2):
        return None
    for i in range(len(temp_matrix1)):
        if i != axis:
            if temp_matrix1[i] != temp_matrix2[i]:
                return None
    return rec(deepcopy(mat1), deepcopy(mat2), axis, 0)


def rec(m1, m2, axis=0, current=0):
    """ recursively function """
    if axis != current:
        return [rec(m1[i], m2[i], axis, current + 1) for i in range(len(m1))]
    m1.extend(m2)
    return m1