#!/usr/bin/env python3
"""Function that loads the pre-made FrozenLakeEnv"""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ the environment"""
    envir = gym.make("FrozenLake-v0", map_name=map_name,
                   desc=desc, is_slippery=is_slippery)
    return envir
