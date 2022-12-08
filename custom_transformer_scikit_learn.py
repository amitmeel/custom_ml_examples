""" 
To write a custom transformer estimator in scikit-learn, you will need to
 create a class that extends the BaseEstimator and TransformerMixin classes,
  and implements the following methods:

__init__: Initialize the model parameters and any other hyperparameters
fit: Fit the model to the training data
transform: Use the fitted model to transform the data
fit_transform: Fit the model to the data and then transform it
Here is an example of a simple custom transformer estimator class:
"""
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y=None):
        # Fit the model to the training data
        pass
    
    def transform(self, X):
        # Use the fitted model to transform the data
        pass
    
    def fit_transform(self, X, y=None):
        # Fit the model to the data and then transform it
        pass

""" 
Once you have implemented the class,
 you can create an instance of the estimator and
  use it like any other scikit-learn estimator, for example:
"""
estimator = CustomTransformer()
X_transformed = estimator.fit_transform(X)
""" 
It is recommended to also implement additional methods such as get_params and set_params
 to make your estimator compatible with scikit-learn's hyperparameter tuning utilities.
 You can find more information and examples of custom transformers in the scikit-learn 
documentation:
 https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator.
"""