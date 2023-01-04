import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class NumericalDataExplorer:
    def __init__(self, df):
        self.df = df
    
    def plot_histograms(self, columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            plt.hist(self.df[col], bins=50)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()
    
    def plot_scatterplots(self, x_col, y_cols=None, hue=None):
        if y_cols is None:
            y_cols = self.df.columns
        for y_col in y_cols:
            sns.scatterplot(x=x_col, y=y_col, hue=hue, data=self.df)
            plt.show()
    
    def plot_boxplots(self, x_col, y_col, hue=None):
        sns.boxplot(x=x_col, y=y_col, hue=hue, data=self.df)
        plt.show()
    
    def plot_correlation_matrix(self):
        corr = self.df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()
        
    def plot_pairplot(self, hue=None):
        sns.pairplot(self.df, hue=hue)
        plt.show()


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

df_x = pd.DataFrame(X, columns=list(data.feature_names))
df_y = pd.DataFrame(y, columns=['target'])
df = pd.concat([df_x, df_y], axis=1)

# Instantiate a DataExplorer object
explorer = NumericalDataExplorer(df)

# Plot histograms of all the columns
explorer.plot_histograms()

# Plot scatterplots of all the columns vs. the 'target' column
explorer.plot_scatterplots('target')

# Plot correlation matrix
explorer.plot_correlation_matrix()