#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import numpy as np
from scipy.integrate import odeint
import random

class DuffingGeneratorClass:
    def duffing_generator(self):
        batches = []
        for _ in range(64):  # Generate 64 batches
            # Randomly generate parameters
            a = random.uniform(-1.5, 1.5)
            b = random.uniform(-1.5, 1.5)
            d = random.uniform(-10, 10)
            gamma = random.uniform(-1.5, 1.5)
            w = random.uniform(0, 10)

            # Time points
            x = torch.linspace(0, 1, 500).view(-1, 1)

            # Duffing differential equation
            def duffing(y, t):
                y0, y1 = y
                dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
                return dydt

            # Initial conditions and solving the ODE
            y0 = [0, 0]
            sol = odeint(duffing, y0, x.view(-1).numpy())
            y = torch.tensor(sol[:, 0], dtype=torch.float32)

            # Create a batch as a tuple
            batch = (y, torch.tensor(d), torch.tensor(a), torch.tensor(b), torch.tensor(gamma), torch.tensor(w), x)
            
            # Append the batch to the list of batches
            batches.append(batch)


        return batches