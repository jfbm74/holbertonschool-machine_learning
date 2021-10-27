#!/usr/bin/env python3
"""Function Q-table"""

import gym
import numpy as np


def q_init(env):
    """ the Q-table as a numpy.ndarray of zeros"""
    a_space_size = env.action_space.n
    s_space_size = env.observation_space.n
    q_table = np.zeros((s_space_size, a_space_size))
    return q_table
