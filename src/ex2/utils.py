import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# @TODO:
# 1. Splitter should provide load save methods for datasets
# 2. FeatureSelector and Classifier should provide load save methods to allow
# these classes to be serialized
class Splitter:
    """
    A class responsible for splitting data into training and testing sets.

    Attributes:
        None

    Methods:
        split(data_df): Splits the input DataFrame into features (X) and target labels (y)
                        and then further splits them into training and testing sets.

    Usage:
        splitter = Splitter()
        X_train, X_test, y_train, y_test = splitter.split(data_df)
    """

    def __init__(self):
        pass

    def split(self, data_df):
        """
        Split the input data into training and testing sets.

        Args:
            data_df (DataFrame): Input data with features and target labels.

        Returns:
            X_train, X_test, y_train, y_test: Split datasets for features and target labels.
        """
        X = data_df[data_df.columns[1:]]
        y = data_df[data_df.columns[0]]
        return train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)


class FeatureSelector:
    """
    A class for performing feature selection and transformation using a pipeline.

    Attributes:
        None

    Methods:
        fit_transform(X): Fits a pipeline to the input data and performs feature selection
                          and transformation.

        transform(X): Applies the previously fitted pipeline to perform feature transformation.

    Usage:
        selector = FeatureSelector()
        X_transformed = selector.fit_transform(X_train)
        X_test_transformed = selector.transform(X_test)
    """

    def __init__(self):
        self.pipe = None

    def fit_transform(self, X):
        """
        Fit a pipeline to the input data and perform feature selection and transformation.

        Args:
            X (DataFrame or array-like): Input data.

        Returns:
            X_transformed: Transformed data after feature selection and transformation.
        """
        self.pipe = Pipeline([('scaler', RobustScaler()), ('var_thresh', VarianceThreshold()), ('pca', PCA())])
        return self.pipe.fit_transform(X)

    def transform(self, X):
        """
        Apply the previously fitted pipeline to perform feature transformation.

        Args:
            X (DataFrame or array-like): Input data.

        Returns:
            X_transformed: Transformed data after feature transformation.
        """
        assert self.pipe is not None
        return self.pipe.transform(X)


class Classifier:
    """
    A class for training and using a Random Forest classifier with hyperparameter tuning.

    Attributes:
        clf: GridSearchCV instance for Random Forest classification.

    Methods:
        fit(X, y): Train the classifier on the input data and labels.

        predict(X): Use the trained classifier to predict labels for input data.

    Usage:
        classifier = Classifier()
        classifier.fit(X_train_transformed, y_train)
        predictions = classifier.predict(X_test_transformed)
    """

    def __init__(self):
        # @TODO: hyperparameter to be tuned and the classifier should be 
        # arguments in the future
        hyperparameters = {
            'n_estimators': [5, 10, 50, 100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 5, 3, 1]
        }
        rf = RandomForestClassifier()
        # @TODO: 1. GridSearch can be replaced with more efficient methods
        # 2. GridSearch can take advantage of multiple cores
        self.clf = GridSearchCV(rf, hyperparameters, cv=10, scoring='f1')

    def fit(self, X, y):
        """
        Train the classifier on the input data and labels.

        Args:
            X (DataFrame or array-like): Input features.
            y (array-like): Target labels.
        """
        self.clf.fit(X, y)

    def predict(self, X):
        """
        Use the trained classifier to predict labels for input data.

        Args:
            X (DataFrame or array-like): Input features.

        Returns:
            predictions: Predicted labels.
        """
        return self.clf.predict(X)
