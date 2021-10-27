#!/usr/bin/env python3
"""Function that has the trained agent play an episode"""

import gym
import numpy as np


def play(env, Q, max_steps=100):
    """ Returns: the total rewards for the episode"""
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, info = env.step(action)
        env.render()

        if done:
            break
    return reward
