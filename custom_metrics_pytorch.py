""" 
To write a custom metric class in PyTorch, 
you will need to create a class that extends the torch.nn.Module class
 and implements the following methods:

__init__: Initialize the metric and any other parameters
forward: Define the forward pass of the metric, computing the value of the metric based on the predicted values and the ground truth values
Here is an example of a simple custom metric class:
"""
import torch
import torch.nn as nn

class CustomMetric(nn.Module):
    def __init__(self, param1=1, param2=2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.sum = 0
        self.count = 0
    
    def forward(self, y_pred, y_true):
        # Compute the value of the metric based on the predicted values and the ground truth values
        self.sum += torch.sum(y_pred - y_true)
        self.count += y_pred.numel()
        return self.sum / self.count

""" 
Once you have implemented the class,
 you can create an instance of the metric and use it like any other PyTorch metric, for example:
"""
metric = CustomMetric()
model = torch.nn.Sequential()
# Add layers to the model
for i in range(100):
    y_pred = model(x)
    metric_value = metric(y_pred, y_true)
""" 
You can find more information and examples of custom metrics
 in the PyTorch documentation: https://pytorch.org/docs/stable/nn.html#writing-custom-layers-and-models.
"""

#########################################################################################################
"""
To write a custom metric in PyTorch,
 you will need to define a function that takes the predicted values and 
 the ground truth values as inputs and returns a tensor representing the metric.

Here is an example of a simple custom metric function:"""

import torch

def custom_metric(y_pred, y_true):
    # Compute the metric between the predicted values and the ground truth values
    metric = torch.mean(torch.square(y_pred - y_true))
    return metric
""" 
Once you have defined the function, you can use it when training a model by passing 
it as an argument to the metric parameter of the torch.nn.Module.eval() method, for example:
"""
model = torch.nn.Sequential()
# Add layers to the model
for i in range(100):
    y_pred = model(x)
    metric_value = model.eval(y_pred, y_true, metric=custom_metric)


"""
You can find more information and
 examples of custom metrics in the PyTorch documentation:
  https://pytorch.org/docs/stable/nn.html#writing-custom-layers-and-models.

"""