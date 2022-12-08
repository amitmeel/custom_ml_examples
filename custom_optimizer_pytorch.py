"""
To write a custom optimizer in PyTorch, 
you will need to create a class that extends the
 torch.optim.Optimizer class and implements the following methods:

__init__: Initialize the optimizer and any other parameters
step: Define the update rules for the weights of the model based on
     the gradients and the learning rate
Here is an example of a simple custom optimizer class:
"""
import torch
import torch.optim as optim

class CustomOptimizer(optim.Optimizer):
    def __init__(self, params, learning_rate=0.01, param1=1, param2=2):
        super().__init__(params, lr=learning_rate)
        self.param1 = param1
        self.param2 = param2
    
    def step(self):
        # Define the update rules for the weights of the model
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(-group["lr"], p.grad.data)

"""
Once you have implemented the class, you can create an instance
 of the optimizer and use it like any other PyTorch optimizer, 
 for example:
"""
model = torch.nn.Sequential()
# Add layers to the model
optimizer = CustomOptimizer(model.parameters())
for i in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)
    optimizer.zero_grad()
    loss.back
