import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, handle_missing='mean', handle_outliers='clip'):
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
    
    def fit(self, X, y=None):
        # Save the column names and indices for missing values
        self.columns_with_missing = X.columns[X.isnull().any()]
        self.indices_with_missing = np.where(X.isnull())[0]
        
        # Save the column names and indices for outliers
        self.columns_with_outliers = []
        self.indices_with_outliers = []
        for col in X.columns:
            lower, upper = np.percentile(X[col], [1, 99])
            indices_outliers = np.where((X[col] < lower) | (X[col] > upper))[0]
            if len(indices_outliers) > 0:
                self.columns_with_outliers.append(col)
                self.indices_with_outliers.extend(indices_outliers)
        
        return self

    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Handle missing values
        if self.handle_missing == 'mean':
            for col in self.columns_with_missing:
                X[col].iloc[self.indices_with_missing] = X[col].mean()
        elif self.handle_missing == 'median':
            for col in self.columns_with_missing:
                X[col].iloc[self.indices_with_missing] = X[col].median()
        elif self.handle_missing == 'mode':
            for col in self.columns_with_missing:
                X[col].iloc[self.indices_with_missing] = X[col].mode()
        
        # Handle outliers
        if self.handle_outliers == 'clip':
            for col in self.columns_with_outliers:
                lower, upper = np.percentile(X[col], [1, 99])
                X[col] = np.clip(X[col], lower, upper)
        elif self.handle_outliers == 'remove':
            X = X.drop(self.indices_with_outliers, axis=0)
        
        return X

#EXAMPLE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

df_x = pd.DataFrame(X, columns=list(data.feature_names))
df_y = pd.DataFrame(y, columns=['target'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

# Instantiate the FeatureEngineering class
fe = FeatureEngineering(handle_missing='mean', handle_outliers='clip')

# Fit the FeatureEngineering class to the train data
fe.fit(X_train, y_train)

# Transform the train and test data
X_train_transformed = fe.transform(X_train, y_train)
X_test_transformed = fe.transform(X_test, y_test)

# Train a classifier on the transformed data
classifier = RandomForestClassifier()
classifier.fit(X_train_transformed, y_train)

# Evaluate the classifier on the transformed test data
print(classifier.score(X_test_transformed, y_test))
