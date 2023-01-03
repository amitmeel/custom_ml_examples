## feature selection using filter methods

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, method='mutual_info_classif', **kwargs):
        self.feature_names = feature_names
        self.method = method
        self.kwargs = kwargs
    
    def fit(self, X, y=None):
        from sklearn.feature_selection import mutual_info_classif
        self.scores_ = mutual_info_classif(X, y, **self.kwargs)
        return self
    
    def transform(self, X, y=None, top_n=10):
        self.selected_indices = np.argpartition(self.scores_, -top_n)[-top_n:]
        return X[:, self.selected_indices]
    
    def get_important_features(self, top_n=10):
        return [self.feature_names[i] for i in self.selected_indices]
    
    def plot_importance_scores_matplotlib(self, top_n=10):
        selected_indices = np.argpartition(self.scores_, -top_n)[-top_n:]
        important_features = [self.feature_names[i] for i in self.selected_indices]
        y_pos = np.arange(len(important_features))

        plt.barh(y_pos, self.scores_[self.selected_indices], align='center')
        plt.yticks(y_pos, important_features, rotation=30)
        plt.xlabel('Importance Score')
        plt.title('Important Features')
        
        for i, v in enumerate(self.scores_[self.selected_indices]):
            plt.text(v + 0.01, i, '{:.3f}'.format(v), va='center', ha='left')
        
        plt.show()


    def plot_importance_scores(self, top_n=10):
        import seaborn as sns
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        # Normalize the scores to use as colors
        norm = colors.Normalize(vmin=self.scores_.min(), vmax=self.scores_.max())
        cmap = cm.Blues

        selected_indices = np.argpartition(self.scores_, -top_n)[-top_n:]
        important_features = [self.feature_names[i] for i in self.selected_indices]
        y_pos = np.arange(len(important_features))

        # Create a barplot showing the feature importance scores
        # ax = sns.barplot(y=important_features, x=self.scores_[self.selected_indices])
        ax = sns.barplot(y=important_features, x=self.scores_[self.selected_indices], palette='Blues')
        ax.set_xlabel('Importance Score')
        ax.set_title('Important Features')
        ax.set_yticklabels(important_features, rotation=30)
        
        # Add text labels with the score values to the plot
        for i, v in enumerate(self.scores_[self.selected_indices]):
            ax.text(v + 0.01, i, '{:.3f}'.format(v), va='center', ha='left')
        
           
        ## Create a second y-axis with a barplot showing the feature importance scores as a color gradient
        # ax2 = ax.twinx()
        # ax2.barh(y_pos, np.ones(len(important_features)), align='center', alpha=0.5, color=cmap(norm(self.scores_[self.selected_indices])))
        # ax2.barh(y_pos, np.ones(len(important_features)), align='center', alpha=0.5, color=cmap(norm(self.scores_[self.selected_indices])))

        # ax2.set_yticks(y_pos)
        # ax2.set_yticklabels(important_features, rotation=30)
        # ax2.invert_yaxis()
        
        # Create a colorbar for the gradient barplot
        # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax2)

        ## Create a second y-axis with a barplot showing the feature importance scores as a color gradient
        # ax2 = ax.twinx()
        # ax2.barh(y_pos, np.ones(len(important_features)), align='center', alpha=0, color=cmap(norm(self.scores_[self.selected_indices])))
        # ax2.set_yticks(y_pos)
        # ax2.set_yticklabels(important_features, rotation=30)
        # ax2.invert_yaxis()
        # ax2.set_yticks([])
        
        # # Create a colorbar for the gradient barplot
        # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax2)
        plt.show()



from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with feature selection and standardization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', FeatureSelector(feature_names, method='mutual_info_classif')),
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Transform the test data using the fitted pipeline
X_test_transformed = pipeline.transform(X_test)

# Print the shape of the transformed test data
print(X_test_transformed.shape)

# Get the most important features
important_features = pipeline.named_steps['selector'].get_important_features(top_n=10)
print(important_features)

# Plot the importance scores of the most important features
pipeline.named_steps['selector'].plot_importance_scores(top_n=10)
