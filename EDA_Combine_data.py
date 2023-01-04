## many things are wrong in this..while running fix it for your usecase


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, df):
        self.df = df
    
    def empty_values(self):
            """
            Method that gives the total number of empty values in each column in a dataframe,
            and also shows a diagram to visualize it.
            """
            # Get the number of empty values in each column
            null_counts = self.df.isnull().sum().sort_values(ascending=False)
            
            # Plot a bar chart to visualize the null counts
            fig = plt.figure(figsize=(10, 5))
            sns.barplot(x=null_counts.index, y=null_counts.values)
            plt.xlabel('Columns')
            plt.ylabel('Number of null values')
            plt.title('Number of null values in each column')
            plt.xticks(rotation=90)
            return fig

        
    def statistical_info(self):
        """
        Method that gives statistical information related to each column,
        whether it is numerical or categorical, in a dataframe.
        """
        # Get the statistical information for each column
        stats = self.df.describe()
        
        # Add the column types to the stats dataframe
        stats['dtype'] = self.df.dtypes
        
        return stats
    
    def column_types(self):
        """
        Method that gives the column type of each column in a dataframe.
        """
        return self.df.dtypes.to_frame()
    
    def duplicates(self):
        """
        Method that gives duplicate information for the dataset.
        """
        # Get the duplicate rows
        duplicates = self.df[self.df.duplicated()]
        
        # Get the count of duplicates
        count = len(duplicates)
        
        return count
    
    def plot_numeric(self, columns, plot_type='hist'):
        """
        Method that plots the distribution of numeric data.
        It can plot the following types:
            - kde (Kernel Density Estimate plot)
            - hist (histogram)
            - box (boxplot)
            - scatter (scatterplot)
            - correlation (correlation plot)
            - pairplot (pairplot)
            - distplot (distplot)
            - violin (violinplot)
        """
        # Validate the plot type
        if plot_type not in ['kde', 'hist', 'box', 'scatter', 'correlation', 'pairplot', 'distplot', 'violin']:
            raise ValueError("Invalid plot type. Choose from 'kde', 'hist', 'box', 'scatter', 'correlation', 'pairplot', 'distplot', 'violin'")
        
        # Validate the columns
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more columns are invalid. Choose from {}".format(self.df.columns))
        
        # Plot the data
        if plot_type == 'kde':
            for col in columns:
                sns.kdeplot(self.df[col])
        elif plot_type == 'hist':
            self.df[columns].hist()
        elif plot_type == 'box':
            self.df[columns].plot(kind='box')
        elif plot_type == 'scatter':
            sns.pairplot(self.df, x_vars=columns, y_vars=columns, kind='scatter')
        elif plot_type == 'correlation':
            sns.pairplot(self.df, x_vars=columns, y_vars=columns, kind='reg')
        elif plot_type == 'pairplot':
            sns.pairplot(self.df[columns])
        elif plot_type == 'distplot':
            for col in columns:
                sns.distplot(self.df[col])
        elif plot_type == 'violin':
            for col in columns:
                sns.violinplot(x=self.df[col])
        plt.show()


    def plot_categorical(self, columns, plot_type='countplot'):
        """
        Method that plots the count distribution of categorical data.
        It can plot the following types:
            - countplot (countplot)
            - barplot (barplot)
            - boxplot (boxplot)
            - violinplot (violinplot)
            - swarmplot (swarmplot)
        """
        # Validate the plot type
        if plot_type not in ['countplot', 'boxplot', 'violinplot', 'swarmplot']:
            raise ValueError("Invalid plot type. Choose from 'countplot', 'barplot', 'boxplot', 'violinplot', 'swarmplot'")
        
        # Validate the columns
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more columns are invalid. Choose from {}".format(self.df.columns))
        
        # Plot the data
        if plot_type == 'countplot':
            for col in columns:
                sns.countplot(x=self.df[col])
        elif plot_type == 'boxenplot':
            sns.catplot(x=columns[0], y=columns[1], kind='boxen', data=self.df)
        elif plot_type == 'boxplot':
            for col in columns:
                sns.boxplot(x=self.df[col])
        elif plot_type == 'violinplot':
            for col in columns:
                sns.violinplot(x=self.df[col])
        elif plot_type == 'swarmplot':
            for col in columns:
                sns.swarmplot(x=self.df[col])
        plt.show()
    
    def time_series(self, columns, frequency):
        """
        Method that plots the time series data by the given frequency.
        It can plot the following frequencies:
            - daily
            - monthly
            - yearly
        """
        # Validate the frequency
        if frequency not in ['daily', 'monthly', 'yearly']:
            raise ValueError("Invalid frequency. Choose from 'daily', 'monthly', 'yearly'")
        
        # Validate the columns
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more columns are invalid. Choose from {}".format(self.df.columns))
        
        # Plot the time series data
        for col in columns:
            plt.figure(figsize=(10, 5))
            if frequency == 'daily':
                self.df[col].plot()
            elif frequency == 'monthly':
                self.df[col].resample('M').mean().plot()
            elif frequency == 'yearly':
                self.df[col].resample('Y').mean().plot()
            plt.ylabel(col)
            plt.title('Time series plot by {}'.format(frequency))
            plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_eda_report(df, report_name):
    """
    Generates an EDA report in HTML format, with plots and statistical information for each column in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to generate the report for.
    report_name (str): The name of the report file. The file will be saved as report_name.html in the current working directory.
    
    Returns:
    None
    """
    # Create an EDA object
    eda = EDA(df)
    
    # Create a list to store the plots and statistics
    report = []
    
    # Add the null value plot
    null_plot = eda.empty_values()
    null_plot.savefig('null_plot.png')
    report.append(('null_plot.png', 'Number of null values in each column'))
    
    # Add the statistical information
    stats = eda.statistical_info()
    report.append(('', 'Statistical information'))
    report.append((stats.to_html(), ''))
    
    # Add the column type information
    col_types = eda.column_types()
    report.append(('', 'Column data types'))
    report.append((col_types.to_html(), ''))
    
    # Add the duplicate count
    duplicates = eda.duplicates()
    report.append(('', 'Number of duplicate rows: {}'.format(duplicates)))
    
    # Add the plots for numeric columns
    num_cols = df.select_dtypes(include='number').columns
    eda.plot_numeric(num_cols, plot_type='hist')
    plt.savefig('numeric_hist.png')
    report.append(('numeric_hist.png', 'Histograms of numeric columns'))
    eda.plot_numeric(num_cols, plot_type='box')
    plt.savefig('numeric_box.png')
    report.append(('numeric_box.png', 'Boxplots of numeric columns'))
    
    # Add the plots for categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    eda.plot_categorical(cat_cols, plot_type='countplot')
    plt.savefig('categorical_count.png')
    report.append(('categorical_count.png', 'Count plots of categorical columns'))
    eda.plot_categorical(cat_cols, plot_type='barplot')
    plt.savefig('categorical_bar.png')
    report.append(('categorical_bar.png', 'Bar plots of categorical columns'))
    
    # Add the correlation plot
    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.savefig('correlation.png')
    report.append(('correlation.png', 'Heatmap of pairwise correlations'))
    
    # Add the outlier plots
    plt.figure()
    sns.boxplot(data=df, orient='h', fliersize=5)
    plt.savefig('outliers.png')
    report.append(('outliers.png', 'Boxplots with outliers marked'))
    
    # Generate the HTML report
    with open(report_name + '.html', 'w') as f:
        f.write('<html><body>')
        for item in report:
            f.write('<h3>{}</h3>'.format(item[1]))
            if item[0]:
                f.write('<img src="{}" style="width:80%">'.format(item[0]))
            else:
                f.write(item[1])
        f.write('</body></html>')

   


df = pd.read_csv(r'C:\Users\0550J9744\Downloads\Rossmann Store Sales Kaggle\train\train.csv')
eda = EDA(df)
generate_eda_report(df, 'eda_report')
print('completed')