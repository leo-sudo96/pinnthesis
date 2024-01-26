#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import numpy as np
from scipy.integrate import odeint
import random
import os
import resource
import sys
def print_memory_usage():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    print(f"Memory usage: {mem} MB")

class DuffingGeneratorClass:
    def duffing_generator(self):
        batches = []
        for _ in range(8):  # Generate 1000 batches
            # Randomly generate parameters
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            d = random.uniform(-2, 5)
            gamma = random.uniform(-5, 5)
            w = random.uniform(0, 5)

            # Time points
            x = torch.linspace(0, 1, 1000).view(-1, 1)

            # Duffing differential equation
            def duffing(y, t):
                y0, y1 = y
                dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
                return dydt

            # Initial conditions and solving the ODE
            y0 = [0, 0]
            sol = odeint(duffing, y0, x.view(-1).numpy())
            #print('solving duffing')
            #print_memory_usage()
            y = torch.tensor(sol[:, 0], dtype=torch.float32)

            # Create a batch as a tuple
            batch = (y, d, a, b, gamma, w, x)

            # Append the batch to the list of batches
            batches.append(batch)

        return batches
