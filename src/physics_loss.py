#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch import autograd
class physics_loss_class:
#physics_loss definition
    def physics_loss(self,model,x_physics,d,a,b,gamma,w):


            # Compute the physics loss by enforcing the differential equation
            yhp = model(x_physics)
            dy_pred = autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]
            d2y_pred = autograd.grad(dy_pred, x_physics, torch.ones_like(dy_pred), create_graph=True)[0]

            physics = d2y_pred + d * dy_pred + a * yhp + b * torch.pow(yhp, 3) - gamma * torch.cos(w * x_physics)
            loss_physics = (1e-4) * torch.mean(physics**2)
            return loss_physics

