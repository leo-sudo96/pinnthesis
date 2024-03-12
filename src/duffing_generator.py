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

class DuffingGeneratorClass:
    def duffing_generator_batch(self, num_batches, x):
        params_list = []  # To store parameters tensors for each batch
        y_physics_list = []  # To store the y_physics tensors for each batch

        for _ in range(num_batches):
            # Randomly generate parameters
            a = random.uniform(-2, 2)
            b = random.uniform(0, 3)
            d = random.uniform(0, 0.5)
            gamma = random.uniform(0, 1.5)
            w = random.uniform(0, 2.5)

            # Duffing differential equation solver setup
            def duffing(y, t):
                y0, y1 = y
                dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
                return dydt

            # Initial conditions and solving the ODE
            y0 = [0, 0]
            sol = odeint(duffing, y0, x.cpu().squeeze().numpy())  # Ensure x is compatible with odeint
            y = torch.tensor(sol[:, 0], dtype=torch.float32).view(-1, 1)  # y_physics for one batch

            y_physics_list.append(y)

            # Handling parameters similarly if needed
            params = torch.tensor([d, a, b, gamma, w], dtype=torch.float32).view(1, -1).repeat(x.size(0), 1)
            params_list.append(params)

        # Option 1: Return lists directly
        # return params_list, y_physics_list

        # Option 2: Stack tensors to create a batch dimension explicitly
        params_tensor = torch.stack(params_list, dim=0)  # Shape: [num_batches, x.size(0), 5]
        y_physics_tensor = torch.stack(y_physics_list, dim=0)  # Shape: [num_batches, x.size(0), 1]

        return params_tensor, y_physics_tensor

