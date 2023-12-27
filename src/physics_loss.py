#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import numpy as np
import torch.autograd as autograd
class physics_loss_class:
   def physics_loss(self, model, x_physics, d_vec, a_vec, b_vec, gamma_vec, w_vec):
       # Initialize total loss
       total_loss = 0.0

       # Loop over each parameter vector
       for d, a, b, gamma, w in zip(d_vec, a_vec, b_vec, gamma_vec, w_vec):
           # Compute the physics loss by enforcing the differential equation
           yhp = model(x_physics)
           dy_pred = autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
           d2y_pred = autograd.grad(dy_pred, x_physics, torch.ones_like(dy_pred), create_graph=True)[0]

           physics = d2y_pred + d * dy_pred + a * yhp + b * torch.pow(yhp, 3) - gamma * torch.cos(w * x_physics)
           loss_physics = (1e-4) * torch.mean(physics**2)

           # Add to total loss
           total_loss += loss_physics

       return total_loss # Return total loss

