""" 
To write a custom metric in TensorFlow,
 you will need to define a function that takes the predicted values 
 and the ground truth values as inputs and returns a tensor representing the metric.

Here is an example of a simple custom metric function:
"""
import tensorflow as tf

def custom_metric(y_pred, y_true):
    # Compute the metric between the predicted values and the ground truth values
    metric = tf.reduce_mean(tf.square(y_pred - y_true))
    return metric

""" 
Once you have defined the function, you can use it when training a model by passing it to the metrics argument 
of the compile method of the tf.keras.Model class, for example:
"""
model = tf.keras.Sequential()
# Add layers to the model
model.compile(optimizer="sgd", loss="mse", metrics=[custom_metric])


########################################################################################

""" 
To write a custom metric class in TensorFlow, 
you will need to create a class that extends the tf.keras.metrics.Metric class 
and implements the following methods:

__init__: Initialize the metric and any other parameters
update_state: Define the forward pass of the metric, updating the internal state of the metric based on the predicted values and the ground truth values
result: Define how to compute the final value of the metric from the updated internal state
Here is an example of a simple custom metric class:

""" 
import tensorflow as tf

class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, param1=1, param2=2, name="custom_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.param1 = param1
        self.param2 = param2
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_pred, y_true, sample_weight=None):
        # Update the internal state of the metric based on the predicted values and the ground truth values
        self.sum.assign_add(tf.reduce_sum(y_pred - y_true))
        self.count.assign_add(tf.cast(tf.size(y_pred), tf.float32))
    
    def result(self):
        # Compute the final value of the metric from the updated internal state
        return self.sum / self.count

""" 
Once you have implemented the class, 
you can create an instance of the metric and use it like any other TensorFlow metric, for example:

""" 
metric = CustomMetric()
model = tf.keras.Sequential()
# Add layers to the model
model.compile(optimizer="sgd", loss="mse", metrics=[metric])

""" 
You can find more information and examples of custom metrics in the TensorFlow 
documentation: https://www.tensorflow.org/guide/keras/custom_metrics#using_a_custom_metric_class.





"""