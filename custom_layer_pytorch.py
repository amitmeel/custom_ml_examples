""" 
To write a custom layer in PyTorch, 
you will need to create a class that extends the torch.nn.Module class
 and implements the following methods:

__init__: Initialize the layer and the weights it contains
forward: Define the forward pass of the layer
Here is an example of a simple custom layer class:
"""

import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self, num_units, param1=1, param2=2):
        super().__init__()
        self.num_units = num_units
        self.param1 = param1
        self.param2 = param2
        self.w = nn.Parameter(torch.randn(num_units, 1))
        self.b = nn.Parameter(torch.randn(num_units, 1))
    
    def forward(self, inputs):
        # Define the forward pass of the layer
        x = torch.mm(inputs, self.w) + self.b
        return torch.relu(x)


""" 
Once you have implemented the class, 
you can create an instance of the layer and use it like any other PyTorch layer, for example:
"""
layer = CustomLayer(num_units=32)
x = torch.ones((1, 10))
y = layer(x)

""" 
You can find more information and examples of custom 
layers in the PyTorch documentation: 
https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html.

"""