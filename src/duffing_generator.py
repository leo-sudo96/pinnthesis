#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import numpy as np
from scipy.integrate import odeint
import random

class DuffingGeneratorClass:
    def __init__(self):
        # Initialization code if needed
        pass

    def duffing_generator(self, x):
        a = random.uniform(0, 1.5)
        b = random.uniform(0, 1.5)
        d = random.uniform(0, 5)
        gamma = random.uniform(0, 1.5)
        w = random.uniform(0, 10)

        def duffing(y, t):
            y0, y1 = y
            dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
            return dydt

        y0 = [0, 0]
        sol = odeint(duffing, y0, x.view(-1).numpy())  # Convert x to a one-dimensional numpy array
        y = torch.tensor(sol[:, 0])
        return [y, d, a, b, gamma, w, x]