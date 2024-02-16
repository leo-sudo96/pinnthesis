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

    def physics_loss(self, model, x_time, x_params,device):
        # Extract the parameters from x_combined
        d = x_params[:, 0]
        a = x_params[:, 1]
        b = x_params[:, 2]
        gamma = x_params[:, 3]
        w = x_params[:, 4]
        x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True).to(device)# sample locations over the problem domain
        
        # Repeat the parameters to match the size of x_physics
        params_repeated = x_params[0, :].repeat(30, 1)
        
        # Combine x_physics with the repeated parameters
        x_physics_combined = torch.cat([x_physics, params_repeated], dim=1)
        x_time = x_physics_combined[:, 0:1]  # Extract the time component
        x_params = x_physics_combined[:, 1:]  # Extract the rest of the parameters
        # Model prediction for physics-based loss
        yhp = model(x_time, x_params)
        def model_output(x_time, x_params):
            return model(x_time, x_params)
        
        def central_difference_second_order(f, x, h=1e-5):
            return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
        # Compute the first derivative using autograd
        dy_pred = torch.autograd.grad(outputs=yhp, inputs=x_physics, grad_outputs=torch.ones_like(yhp), create_graph=True)[0]

        # Compute the first derivative using autograd
        d2y_pred = torch.autograd.grad(outputs=yhp, inputs=x_physics, grad_outputs=torch.ones_like(yhp), create_graph=True)[0]
        
        # Define a wrapper for the model output
        def model_output(x):
            return model(x, x_params)

        # Compute the second derivative using the custom method
        d2y_dx2 = central_difference_second_order(model_output, x_physics, h=1e-5)

        # Calculate physics-based loss
        physics = d2y_pred + d * dy_pred + a * yhp + b * torch.pow(yhp, 3) - gamma * torch.cos(w * x_physics)
        loss_physics = (1e-3) * torch.mean(physics**2)

        return loss_physics

  
    