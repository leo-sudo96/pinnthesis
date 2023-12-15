#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class duffing_solver:
    # Define a function to solve the Duffing equation numerically
    def solve_duffing(d, a, b, gamma, w, x):
        def duffing(y, t):
            y0, y1 = y
            dydt = [y1, -d * y1 - a * y0 - b * y0**3 + gamma * np.cos(w * t)]
            return dydt

        y0 = [0, 0]  
        sol = odeint(duffing, y0, x.view(-1).numpy())  # Convert x to a one-dimensional numpy array
        y = torch.tensor(sol[:, 0])
        return y

