#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import torch
from scipy.integrate import odeint
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from torch import nn, autograd
import random
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import os


# In[2]:


# Define the device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
current_directory = os.getcwd()


# In[3]:


def plot_result(x, y_tensor_squeezed, x_data, y_data, yh, xp=None, physics_params=None, model=None):
    """
    Plots training data, ground truth, model predictions, and optionally physics predictions.

    Parameters:
    - x: The full range of x values for ground truth data.
    - y_tensor_squeezed: Ground truth data corresponding to x.
    - x_data: X values of the training data.
    - y_data: Y values of the training data (observed outputs).
    - yh: Model predictions corresponding to x_data.
    - xp: Optional, additional x points for physics-based predictions.
    - physics_params: Optional, physics parameters for model if physics predictions are desired.
    - model: The trained model, required if physics_params and xp are provided.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(x_data.detach().numpy(), y_data.detach().numpy(), color="tab:orange", label="Training data")
    
    # Plot ground truth data
    plt.plot(x.detach().numpy(), y_tensor_squeezed.detach().numpy(), 'r-', label='Ground Truth Data')
    
    # Plot model predictions
    plt.plot(x_data.detach().numpy(), yh.detach().numpy(), 'b--', label='Model Predictions')
    
    # Optionally, plot physics-based predictions
    if xp is not None and physics_params is not None and model is not None:
        # Assuming model can directly use xp and physics_params for prediction
        # You might need to adjust this call based on how your model uses physics_params
        yhp = model(torch.cat((xp, torch.tensor(physics_params).repeat(len(xp), 1)), dim=1)).detach()
        plt.plot(xp.detach().numpy(), yhp.numpy(), 'g:', label='Physics Predictions')
    
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[4]:


class OscillationDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):#, boundary_conditions
        self.data = data
        self.targets = targets
        #self.boundary_conditions = boundary_conditions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        #boundary_condition = self.boundary_conditions[idx]
        return sample, target#, boundary_condition


# In[5]:


def generate_data_with_boundaries(num_samples, x):
    data = []
    #boundary_values = []
    f_boundary_values = []  # This might be used differently, depending on what exactly you need
    
    # Ensure x is a numpy array for compatibility with odeint
    x_np = x.numpy() if isinstance(x, torch.Tensor) else x

    for _ in tqdm.tqdm(range(num_samples)):
        # Randomly choose parameters for each sample
        a = random.uniform(-2, 2)
        b = random.uniform(0, 3)
        d = random.uniform(0, 0.5)
        gamma = random.uniform(0, 1.5)
        omega = random.uniform(0, 2.5)

        # Define initial conditions
        #y0 = [random.uniform(-1, 1), random.uniform(-1, 1)]
        y0 = [0, 0]
        # Solve the Duffing oscillator equation
        # Local definition of the differential equation with captured parameters
        
        def duffing(y, t, a=a, b=b, d=d, gamma=gamma, omega=omega):
            x, x_dot = y
            d2x_dt2 = -d * x_dot - a * x - b * x**3 + gamma * np.cos(omega * t)
            return [x_dot, d2x_dt2]

        sol = odeint(duffing, y0, x_np, args=(a, b, d, gamma, omega))
        x_t = sol[:, 0]  # Solution x(t)
       
        # Compile the parameters and outputs to form the dataset
        for xi, xti in zip(x_np, x_t):
            data.append([xi, a, b, d, gamma, omega, xti])
            
        
    return np.array(data)#, np.array(boundary_values)


# In[6]:


#data, boundary_values = generate_data_with_boundaries(num_samples=10000, x=np.linspace(0, 10, 500))
data= generate_data_with_boundaries(num_samples=1000000, x=np.linspace(0, 10, 5000))
# Convert the generated data to PyTorch tensors
X = torch.tensor(data[:, :-1], dtype=torch.float32)  # Input features: [x_i, a, b, d, gamma, omega]
Y = torch.tensor(data[:, -1], dtype=torch.float32).view(-1, 1)  # Targets: x(t)


# In[7]:


# Assuming data generation and conversion to tensors has been done as previously described
X_tensor = torch.tensor(data[:, :-1], dtype=torch.float32)#.requires_grad_(True)  # Features: [x_i, a, b, d, gamma, omega]
Y_tensor = torch.tensor(data[:, -1], dtype=torch.float32).view(-1, 1)  # Targets: x(t_i)
# Set up the boundary conditions
X_BOUNDARY = [0.0]  # boundary condition coordinate
F_BOUNDARY = [0.0]  # boundary condition value

x_boundary = torch.tensor([X_BOUNDARY]).requires_grad_(True)
f_boundary = torch.tensor([F_BOUNDARY]).requires_grad_(True)

# Assuming boundary_conditions is prepared alongside data and targets
oscillation_dataset = OscillationDataset(X_tensor, Y_tensor)


# In[8]:


# Assuming boundary_conditions is prepared alongside data and targets
oscillation_dataset = OscillationDataset(X_tensor, Y_tensor)#, boundary_conditions_tensor


oscillation_dataloader = DataLoader(oscillation_dataset, batch_size=2048, shuffle=True, num_workers=4)#, collate_fn=custom_collate


# In[9]:


# Define a class to create a fully connected neural network
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS-1)]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        # Assuming your linear layer is named `layer`
        input_size = self.fce.in_features
        # print("Input size of the linear layer: ", input_size)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# In[10]:


N_INPUT = 6 # [x_i, a, b, d, gamma, omega]
N_HIDDEN = 128
N_OUTPUT = 1  # x(t)
N_LAYERS = 4
epochs = 100000
model = FCN(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model = model.to(device)
tensor_list = [X_tensor, Y_tensor, x_boundary, f_boundary]  # Your tensors
tensor_list = [t.to(device) for t in tensor_list]


# In[11]:


initial_lr = 0.01
num_warmup_steps = 10
num_total_steps = 100000
decay_rate = 0.1
decay_steps = 100

def lr_lambda(current_step: int):
    if current_step < num_warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        # Exponential decay
        return decay_rate ** ((current_step - num_warmup_steps) // decay_steps)

#Define the scheduler
scheduler = LambdaLR(optimizer, lr_lambda)


# In[12]:


class PhysicsInformedLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, yhp, data, d, a, b, gamma, omega):
        # Forward pass computations
        ctx.save_for_backward(data, d, a, b, gamma, omega, yhp)
        # Loss computation
        return loss_physics

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        data, d, a, b, gamma, omega, yhp = ctx.saved_tensors
        
        # Example gradient computations (conceptual and simplified)
        # These would need to be replaced with the actual gradients based on your loss function
        grad_yhp = torch.autograd.grad(physics_loss, yhp, grad_outputs=grad_output, create_graph=True)[0]
        grad_data = torch.autograd.grad(physics_loss, data, grad_outputs=grad_output, create_graph=True)[0]

        # Assume gradients w.r.t. other parameters are not required
        return grad_yhp, grad_data, None, None, None, None, None


# In[13]:


for epoch in tqdm.tqdm(range(epochs)):
    for batch_idx, (sample, targets) in enumerate(oscillation_dataloader):#, batch_boundary_conditions
        optimizer.zero_grad()
         # Set requires_grad to True for data
        sample.requires_grad_(True)
        data = sample.to(device)
        
        targets = targets.to(device)
        # Model prediction for the full dataset
        yh = model(data)
            # Data loss (comparing model output with true data)
        y_data = Y_tensor  # True output values from your dataset
        loss_data = torch.mean((yh - targets)**2)
        
        # Extract domain (time) and parameters from X_tensor
        x_domain = data[:, 0].view(-1, 1).requires_grad_(True)
        params = data[:, 1:]  # Parameters: a, b, d, gamma, omega
        params = params.to(device)
        a, b, d, gamma, omega = params.t()  # Transpose for convenience in calculations
    
        yhp = model(data)
        dy_pred = torch.autograd.grad(yhp, data, torch.ones_like(yhp), create_graph=True)[0]
        d2y_pred = torch.autograd.grad(dy_pred, data, torch.ones_like(dy_pred), create_graph=True)[0]

        physics_loss = d2y_pred + d.unsqueeze(1) * dy_pred + a.unsqueeze(1) * yhp + b.unsqueeze(1) * torch.pow(yhp, 3) - gamma.unsqueeze(1) * torch.cos(omega.unsqueeze(1) * data)
        loss_physics = (1e-4) * torch.mean(torch.square(physics_loss))
       
        x_boundary = torch.tensor([X_BOUNDARY]).requires_grad_(True)
        x_boundary =  x_boundary.repeat(params.size(0),1)
        x_boundary = x_boundary.to(device)
        x_boundary = torch.cat([x_boundary,params],dim=1)
        f_boundary = torch.tensor([F_BOUNDARY]).requires_grad_(True)
        f_boundary = f_boundary.to(device)
        yh_boundary = model(x_boundary)
        boundary = yh_boundary - f_boundary
        loss_boundary = (1e-6) * torch.mean(boundary**2)
    
        
        # Combined loss
        total_loss = loss_physics + loss_data + loss_boundary 
    
        total_loss.backward()
        optimizer.step()
    #scheduler.step()
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Total Loss: {total_loss.item()}')

    if  epoch % 10 == 0:
        model_save_path = os.path.join(current_directory, "/models/fcn_pinn_10sec_higher_schedulerparam_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
            
    if total_loss < 1e-7:
        model_save_path = os.path.join(current_directory, "/models/fcn_pinn_10sec_higher_schedulerparam_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
                


# In[ ]:


# Plotting results
        plot_result(x=X_tensor[:, 0],  # Domain (time points)
                    y_tensor_squeezed=Y_tensor,  # Ground truth data
                    x_data=X_tensor[:, 0],  # Same as domain for plotting
                    y_data=Y_tensor,  # Ground truth data for scatter plot
                    yh= yh,  # Model predictions
                    xp=None,  # Additional physics-based prediction points, if applicable
                    physics_params=None,  # Physics parameters, if needed for additional predictions
                    model= yhp)  # Model, if additional physics-based predictions are to be plotted    


# In[ ]:




