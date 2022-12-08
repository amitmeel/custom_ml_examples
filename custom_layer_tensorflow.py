""" 
To write a custom layer in TensorFlow, you will need to create a class that
 extends the tf.keras.layers.Layer class and implements the following methods:

__init__: Initialize the layer and the weights it contains
build: Initialize the weights of the layer
call: Define the forward pass of the layer
Here is an example of a simple custom layer class:
"""

import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, param1=1, param2=2):
        super().__init__()
        self.num_units = num_units
        self.param1 = param1
        self.param2 = param2
    
    def build(self, input_shape):
        # Initialize the weights of the layer
        self.w = self.add_weight(shape=(input_shape[-1], self.num_units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.num_units,),
                                 initializer="random_normal",
                                 trainable=True)
    
    def call(self, inputs):
        # Define the forward pass of the layer
        x = tf.matmul(inputs, self.w) + self.b
        return tf.nn.relu(x)

""" 
Once you have implemented the class,
 you can create an instance of the layer and
  use it like any other TensorFlow layer, for example:"""

layer = CustomLayer(num_units=32)
x = tf.ones((1, 10))
y = layer(x)

""" 
You can find more information and examples of custom layers in the
 TensorFlow documentation: https://www.tensorflow.org/guide/keras/custom_layers_and_models."""