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
    



class physics_loss_class:
    def physics_loss(self, model, params_physics,x_physics):
        #combined_input.requires_grad_(True)
        current_params = params_physics
        # Ensure the first dimension (time) is used for differentiation
        #time = combined_input[:, 0].unsqueeze(1)
        #time.requires_grad_(True)
        x = x_physics.view(-1,1).requires_grad_(True)
        # Predict model output using combined input
        yhp = model(x,params_physics)
        
        # Compute the first derivative with respect to time
        dy_pred = torch.autograd.grad(outputs=yhp, inputs=x,
                                      grad_outputs=torch.ones_like(yhp),
                                      create_graph=True, allow_unused=True)[0]

        # Check if dy_pred is None and handle it
        if dy_pred is None:
            raise ValueError("First derivative (dy_pred) could not be computed. Check model dependencies on time.")

        # Compute the second derivative
        d2y_pred = torch.autograd.grad(outputs=dy_pred, inputs=x,
                                       grad_outputs=torch.ones_like(dy_pred),
                                       create_graph=True, allow_unused=True)[0]

        # Check if d2y_pred is None and handle it
        if d2y_pred is None:
            raise ValueError("Second derivative (d2y_pred) could not be computed. Check model dependencies on time.")

        
        # Compute physics-based loss using dydt, d2ydt2, and any necessary physical equations

        # Since parameters are repeated for each time step, every row in the first 5 columns contains the parameters
        # for the corresponding time step, no need to reshape or repeat them again
        d, a, b, gamma, w = current_params[:, 0], current_params[:, 1], current_params[:, 2], current_params[:, 3], current_params[:, 4]

        # Calculate the physics-based component of the loss
        # Ensure d2y_pred and other derivatives are not None before proceeding
        if d2y_pred is not None:
            physics_loss = d2y_pred + d * dy_pred + a * yhp + b * yhp.pow(3) - gamma * torch.cos(w * x)
            loss_physics = (1e-4) * torch.mean(physics_loss.pow(2))
        else:
            raise RuntimeError("Failed to compute the second derivative. Check the combined_input and model structure.")

        return loss_physics
  
    