#!/usr/bin/env python3
"""Function that calculates the weighted moving average """


def moving_average(data, beta):
    """Function that calculates the weighted moving average """
    V = [0]
    for i in range(len(data)):
        V.append((beta * V[i]) + ((1 - beta) * data[i]))
    move = []
    for i in range(1, len(V)):
        move.append(V[i] / (1 - (beta ** i)))
    return move
