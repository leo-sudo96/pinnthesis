
#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.rnn = nn.RNN(N_INPUT, N_HIDDEN, N_LAYERS, batch_first=True)
        self.fc = nn.Linear(N_HIDDEN, N_OUTPUT)

        input_size = self.fc.in_features
        print("Input size of the linear layer:", input_size)
        # add parameters a, b, d, gamma, w, x-time

    def forward(self, x):
        # x is expected to be of shape (batch_size, sequence_length, N_INPUT)
        with torch.backends.cudnn.flags(enabled=False):
            output, _ = self.rnn(x)
        #print("model computation")
        #print_memory_usage()
        # Check if the output is 2D or 3D and handle accordingly
        if output.dim() == 3:
            # If 3D, select the last time step
            output = output[:, -1, :]
        elif output.dim() == 2:
            # If already 2D, use as is
            output = output

        output = self.fc(output)
        return output