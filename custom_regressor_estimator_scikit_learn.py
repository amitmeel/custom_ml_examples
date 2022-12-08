""" 
o write a custom regressor estimator in scikit-learn,
 you will need to create a class that extends the BaseEstimator and 
 RegressorMixin classes, and implements the following methods:

__init__: Initialize the model parameters and any other hyperparameters
fit: Fit the model to the training data
predict: Use the fitted model to make predictions on new data
score: Calculate the model's performance on the given data and labels
Here is an example of a simple custom regressor estimator class
"""
from sklearn.base import BaseEstimator, RegressorMixin

class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        # Fit the model to the training data
        pass
    
    def predict(self, X):
        # Use the fitted model to make predictions on new data
        pass
    
    def score(self, X, y):
        # Calculate the model's performance on the given data and labels
        pass
""" 
Once you have implemented the class, 
you can create an instance of the estimator and use it like any other scikit-learn estimator, 
for example:
"""
estimator = CustomRegressor()
estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)
score = estimator.score(X_test, y_test)
"""
It is recommended to also implement additional methods such as get_params and 
set_params to make your estimator compatible with scikit-learn's hyperparameter 
tuning utilities. You can find more information and examples of custom 
regressors in the scikit-learn documentation:
 https://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator.
"""




Try a"""