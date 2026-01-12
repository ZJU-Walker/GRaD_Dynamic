"""Model initialization utilities."""
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    """Initialize module weights and biases."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name):
    """Get activation function by name."""
    if activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError(f'Activation func {activation_name} not defined')