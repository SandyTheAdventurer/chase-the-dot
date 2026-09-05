import torch
from torch import nn

def mlp(in_dim, hidden_sizes, out_dim, activation=nn.ReLU):
    """
    Constructs a Multi-Layer Perceptron (MLP) with the given dimensions.
    """
    layers = []
    current_dim = in_dim

    for size in hidden_sizes:
        layers.append(nn.Linear(current_dim, size))
        layers.append(activation())
        current_dim = size

    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)
