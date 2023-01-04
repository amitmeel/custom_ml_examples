import seaborn as sns
import matplotlib.pyplot as plt

class CategoricalEDA:
    def __init__(self, df, target=None):
        self.df = df
        self.target = target
    
    def plot_countplot(self, column, hue=None):
        """This plot displays the frequency distribution of a categorical variable.
        It is similar to a histogram, but for categorical data
        """
        sns.countplot(x=column, hue=hue, data=self.df)
        plt.show()
    
    def plot_barplot(self, column, hue=None):
        """
        This plot displays the mean (or other summary statistic) of a numerical
        variable for each category of a categorical variable.
        """
        sns.barplot(x=column, y=self.target, hue=hue, data=self.df)
        plt.show()
    
    def plot_boxplot(self, column, hue=None):
        """
        This plot displays the distribution of a numerical variable for each
        category of a categorical variable. It shows the median, interquartile range, and range of the data.
        """
        sns.boxplot(x=column, y=self.target, hue=hue, data=self.df)
        plt.show()
    
    def plot_violinplot(self, column, hue=None):
        """
        This plot is similar to a boxplot, but it also shows the kernel
        density estimate of the data.
        """
        sns.violinplot(x=column, y=self.target, hue=hue, data=self.df)
        plt.show()
    
    def plot_swarmplot(self, column, hue=None):
        """
        This plot is similar to a scatterplot, but it displays the distribution 
        of a numerical variable for each category of a categorical variable.
        """
        sns.swarmplot(x=column, y=self.target, hue=hue, data=self.df)
        plt.show()

# >>> Example:
# Create an instance of the CategoricalEDA class
eda = CategoricalEDA(df, target='target_column')

# Plot the frequency distribution of a categorical feature
eda.plot_countplot('feature_1')

# Plot the relationship between a categorical feature and the target variable
eda.plot_barplot('feature_2')
