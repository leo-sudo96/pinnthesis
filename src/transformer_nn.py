#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
    


class TransformerModel(nn.Module):
    def __init__(self, n_time_features, n_param_features, n_hidden, n_layers, n_heads):
        super(TransformerModel, self).__init__()
        # Embedding layers with non-linear activation
        self.time_embedding = nn.Sequential(
            nn.Linear(n_time_features, n_hidden),
            nn.ReLU()
        )
        self.param_embedding = nn.Sequential(
            nn.Linear(n_param_features, n_hidden),
            nn.ReLU()
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layers - a small feed-forward network
        self.output_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),  # Reducing the dimension
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.ReLU(),
            nn.Linear(n_hidden // 4, 1)  # Final output layer
        )

    def forward(self, x_time, x_params):
        time_embedded = self.time_embedding(x_time)
        params_embedded = self.param_embedding(x_params)

        # Combine embeddings
        x_combined = time_embedded + params_embedded  # or any other combination

        # Transformer Encoder
        output = self.transformer_encoder(x_combined)

        # Final output
        output = self.output_layer(output)
        return output
