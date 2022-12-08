"""To write a custom optimizer in TensorFlow, you will need to
 create a class that extends the tf.keras.optimizers.Optimizer 
 class and implements the following methods:

__init__: Initialize the optimizer and any other parameters
get_updates: Define the update rules for the weights of the
     model based on the gradients and the learning rate
Here is an example of a simple custom optimizer class:
"""
import tensorflow as tf

class CustomOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, param1=1, param2=2, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.param1 = param1
        self.param2 = param2
    
    def get_updates(self, loss, params):
        # Define the update rules for the weights of the model
        grads = self.get_gradients(loss, params)
        updates = []
        for param, grad in zip(params, grads):
            new_param = param - self.learning_rate * grad
            updates.append(param.assign(new_param))
        return updates
""" 
Once you have implemented the class,
 you can create an instance of the optimizer 
 and use it like any other TensorFlow optimizer, for example:
"""
optimizer = CustomOptimizer()
model = tf.keras.Sequential()
# Add layers to the model
model.compile(optimizer=optimizer, loss="mse")

""" 
You can find more information and examples of custom optimizers 
in the TensorFlow documentation: 
https://www.tensorflow.org/guide/keras/custom_layers_and_models#using_a_custom_optimizer.
"""