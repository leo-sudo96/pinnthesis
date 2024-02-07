#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import numpy as np
import torch.autograd as autograd
import os
import resource
import sys
def print_memory_usage():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    print(f"Memory usage: {mem} MB")
    

import torch.autograd as autograd

class physics_loss_class:
    def physics_loss(self, model, batch, x_data, y_data):
        # Unpack the batch
        if x_data.dim() == 2:
            x_data = x_data.unsqueeze(1)  # Add a sequence length dimension

        y, d, a, b, gamma, w, x_physics = batch

        # Ensure that x_physics requires gradient
        x_physics.requires_grad_(True)

        # Compute the physics-based loss
        loss_physics = self.compute_physics_loss(model, x_physics, d, a, b, gamma, w)

        # Compute the MSE loss
        mse_loss = self.compute_mse_loss(model, x_data, y_data)

        # Combine losses
        total_loss = loss_physics + mse_loss

        return total_loss

    def compute_physics_loss(self, model, x_physics, d, a, b, gamma, w):
        # Set up the physics loss training locations
        x_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
        x_physics = x_physics.to('cuda')
        model = model.to('cuda')
        # Model prediction for physics-based loss
        yhp = model(x_physics)

        # Compute derivatives
        dy_pred = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
        d2y_pred = torch.autograd.grad(dy_pred, x_physics, torch.ones_like(dy_pred), create_graph=True)[0]

        # Calculate physics-based loss
        physics = d2y_pred + d * dy_pred + a * yhp + b * torch.pow(yhp, 3) - gamma * torch.cos(w * x_physics)
        loss_physics = (1e-3) * torch.mean(physics**2)

        return loss_physics


    def compute_mse_loss(self, model, x_data, y_data):
        # Check the dimension of x_data and reshape if necessary
        if x_data.dim() == 2:
            x_data = x_data.unsqueeze(1)  # Add a sequence length dimension
    
        # Compute the MSE loss
        yh = model(x_data)
        mse_loss = torch.mean((yh - y_data) ** 2)
    
        return mse_loss