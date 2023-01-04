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
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(10, 8))  # decrease figure size
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, cmap=cmap, vmax=1, center=0, annot=True,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},# decrease font size 
                    annot_kws={'fontsize': 5},
                    # mask=mask
                    )  
        # plt.xticks(rotation=45)
        # plt.yticks(rotation=45)
        plt.subplots_adjust(hspace=0.8, wspace=0.8)  # add padding between rows and columns
        plt.show()
        
    def plot_pairplot(self, hue=None):
        sns.pairplot(self.df, hue=hue)
        plt.show()

    def plot_kde(self, column, shade=True, color='b', bw=0.5, gridsize=100):
        sns.kdeplot(self.df[column], shade=shade, color=color, bw=bw, gridsize=gridsize)
        plt.show()

    def plot_kde_all(self):
        numerical_columns = self.df.select_dtypes(include=['float', 'int']).columns
        for column in numerical_columns:
            self.plot_kde(column)

    def plot_hist_kde(self, column, kde=True, hist=True, kde_kws=None, hist_kws=None, bins='auto'):
            sns.displot(self.df[column], kde=kde, kde_kws=kde_kws, bins=bins)
            plt.show()
    
    def plot_displot(self):
        # histogram and kde together
        numerical_columns = self.df.select_dtypes(include=['float', 'int']).columns
        for column in numerical_columns:
            self.plot_hist_kde(column)


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
# explorer.plot_histograms()

# Plot scatterplots of all the columns vs. the 'target' column
# explorer.plot_scatterplots('target')

# Plot correlation matrix
# explorer.plot_correlation_matrix()

# Plot KDE plot
# explorer.plot_kde_all()

# plot displot
explorer.plot_displot()