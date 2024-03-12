import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, n_time_features, n_param_features, n_hidden, n_layers, negative_slope=0.01):
        super(FullyConnectedNet, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(n_time_features + n_param_features, n_hidden)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))

        # Output layer
        self.output_layer = nn.Linear(n_hidden, 1)
        
        # Initialize LeakyReLU with the specified negative slope
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x, params):
        # Concatenate time and parameter features
        x_combined = torch.cat((x, params), dim=1)

        # Input layer with LeakyReLU
        x = self.leaky_relu(self.input_layer(x_combined))

        # Hidden layers with LeakyReLU
        for hidden_layer in self.hidden_layers:
            x = self.leaky_relu(hidden_layer(x))

        # Output layer
        x = self.output_layer(x)
        return x
       
class FullyConnectedNetTanh(nn.Module):
    def __init__(self, n_time_features, n_param_features, n_hidden, n_layers):
        super(FullyConnectedNetTanh, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(n_time_features + n_param_features, n_hidden)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))

        # Output layer
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x, params):
        # Concatenate time and parameter features
        x_combined = torch.cat((x, params), dim=1)

        # Input layer with Tanh
        x = torch.tanh(self.input_layer(x_combined))

        # Hidden layers with Tanh
        for hidden_layer in self.hidden_layers:
            x = torch.tanh(hidden_layer(x))

        # Output layer
        x = self.output_layer(x)
        return x


class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh

        # Adjust the first layer to take concatenated input of x and a
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT * 2, N_HIDDEN),  # N_INPUT is now doubled
            activation()
        )

        self.fch = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS-1)]
        )

        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x, a):
        # Concatenate x and a along the last dimension
        combined_input = torch.cat((x, a), dim=-1)
        
        x = self.fcs(combined_input)
        x = self.fch(x)
        x = self.fce(x)
        return x



class AdaptedFCN(nn.Module):
    def __init__(self, n_time_features, n_param_features, n_hidden, n_layers):
        super(AdaptedFCN, self).__init__()
        
        # Number of input features is the sum of time features and parameter features
        self.input_features = n_time_features + n_param_features
        
        # Input layer takes combined time and parameters
        self.input_layer = nn.Linear(self.input_features, n_hidden)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
        
        # Output layer
        self.output_layer = nn.Linear(n_hidden, 1)
        
        # Activation function
        self.activation = nn.Tanh()  # Using Tanh as an example; adjust as needed

    def forward(self, x, params):
        # Concatenate time and parameter features
        x_combined = torch.cat((x, params), dim=1)
        
        # Input layer with activation
        x = self.activation(self.input_layer(x_combined))
        
        # Hidden layers with activation
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output layer without activation (assuming a regression task)
        x = self.output_layer(x)
        return x