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


import torch
import random
import numpy as np
from scipy.integrate import odeint
class DuffingGeneratorClass:
        def duffing_generator_batch(self,num_batches):
            batches = []
            for _ in range(num_batches):
                # Randomly generate parameters
                a = random.uniform(-1, 1)
                b = random.uniform(-0.5,0.5 )
                d = random.uniform(0, 10)
                gamma = random.uniform(0, 5)
                w = random.uniform(0, 2)
        
                # Time points
                x = torch.linspace(0, 10, 100).view(-1, 1)
        
                # Duffing differential equation
                def duffing(y, t):
                    y0, y1 = y
                    dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
                    return dydt
        
                # Initial conditions and solving the ODE
                y0 = [0, 0]
                sol = odeint(duffing, y0, x.view(-1).numpy())
                y = torch.tensor(sol[:, 0], dtype=torch.float32)
        
                # Combine parameters with x for each time step
                params = torch.tensor([d, a, b, gamma, w], dtype=torch.float32).view(1, -1)
                params = params.repeat(x.size(0), 1)
                x_combined = torch.cat((x, params), dim=1)
        
                # Create a batch as a tuple (combined input, output)
                batch = (x_combined, y)
        
                # Append the batch to the list of batches
                batches.append(batch)
        
            # Return batches
            return batches