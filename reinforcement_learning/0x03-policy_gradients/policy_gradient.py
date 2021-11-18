#!/usr/bin/env python3
"""Policy and policy_gradient"""

import numpy as np


def policy(matrix, weight):
    """Function that resolves with a weight"""
    z = np.dot(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """Function that resolves the Monte-Carlo policy
    gradient """
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    dsoftmax = softmax[action, :]
    dlog = dsoftmax / P[0, action]
    grad = state.T.dot(dlog[None, :])
    return action, grad
