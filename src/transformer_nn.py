#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)

        # Directly create TransformerEncoderLayer instances
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.output(x)
        return x