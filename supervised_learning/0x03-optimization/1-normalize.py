#!/usr/bin/env python3
"""Function that normalizes a matrix"""

import numpy as np


def normalize(X, m, s):
    """Function that normalizes a matrix"""
    return (X - m) / s
