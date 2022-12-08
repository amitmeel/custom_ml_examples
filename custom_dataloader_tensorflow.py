"""
To write a custom data loader in TensorFlow,
 you will need to create a class that extends the tf.data.Dataset 
 class and implements the following methods:

__init__: Initialize the data source and any other parameters
__len__: Return the total number of samples in the dataset
__getitem__: Return the sample at the specified index
map: Apply a transformation to the data
batch: Group the data into batches
shuffle: Shuffle the data
repeat: Repeat the data 

Here is an example of a simple custom data loader class:
"""
import tensorflow as tf

class CustomDataLoader(tf.data.Dataset):
    def __init__(self, data_source, param1=1, param2=2):
        self.data_source = data_source
        self.param1 = param1
        self.param2 = param2
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_source)
    
    def __getitem__(self, index):
        # Return the sample at the specified index
        return self.data_source[index]
    
    def map(self, func):
        # Apply a transformation to the data
        return self.data_source.map(func)
    
    def batch(self, batch_size):
        # Group the data into batches
        return self.data_source.batch(batch_size)
    
    def shuffle(self, buffer_size):
        # Shuffle the data
        return self.data_source.shuffle(buffer_size)
    
    def repeat(self):
        # Repeat the data
        return self.data_source.repeat()

""" 
Once you have implemented the class, 
you can create an instance of the data loader and
 use it like any other TensorFlow data loader, for example:
"""
data_loader = CustomDataLoader(data_source)
dataset = data_loader.map(lambda x: x * 2).batch(32).shuffle(1024).repeat()
for batch in dataset:
    # Process the batch
    pass
""" 
You can find more information and 
examples of custom data loaders in the TensorFlow documentation: https://www.tensorflow.org/guide/data.
"""