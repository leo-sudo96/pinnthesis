#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import os
import resource
import sys
def print_memory_usage():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    print(f"Memory usage: {mem} MB")
    
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
#1. The `RNN` class replaces the `FCN` class.
#2. The `rnn` module is an instance of `nn.RNN`, which takes the input size (`N_INPUT`), hidden size (`N_HIDDEN`), and number of layers #(`N_LAYERS`) as parameters. The `batch_first=True` argument ensures that the input shape is `(batch_size, sequence_length, N_INPUT)` for #efficient batch processing.
#3. The `fc` module is a linear layer that takes the hidden state of the RNN as input and produces the final output.
#4. The `forward` method performs the forward pass. It passes the input `x` through the RNN module, extracts the hidden state from the last #time step, and passes it through the linear layer to get the output.

#Remember to update the input and output dimensions (`N_INPUT` and `N_OUTPUT`) based on your specific task requirements. Additionally, you may #need to modify the loss function and data preprocessing to handle the sequential nature of the data when training the RNN.