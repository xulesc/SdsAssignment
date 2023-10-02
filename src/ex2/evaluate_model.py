import sys
from warnings import simplefilter

import numpy as np
import pandas as pd
import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score

# @TODO: all prints should be handled through logging. this will allows us to
# build process monitoring infrastructure in the future
class Evaluator:
    """
    A class responsible for evaluating a machine learning pipeline.

    Attributes:
        infile_path (str): The path to the input CSV file.

    Methods:
        evaluate(): Orchestrates the entire evaluation process, including data loading,
                    preprocessing, model training, and performance evaluation.

    Usage:
        evaluator = Evaluator('data/raw/SemgHandGenderCh2.csv')
        evaluation_results = evaluator.evaluate()
    """

    def __init__(self, infile_path):
        """
        Initialize the Evaluator instance.

        Args:
            infile_path (str): The path to the input CSV file.
        """
        # Ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        # Load raw data
        self.data_df = pd.read_csv(infile_path)
        print(f"Raw data size {self.data_df.shape}")

    def evaluate(self):
        """
        Orchestrates the evaluation process.

        Returns:
            results_df (DataFrame): A DataFrame containing evaluation results.
        """
        # Train-test split
        print('Running Splitter')
        X_train, X_test, y_train, y_test = utils.Splitter().split(self.data_df)
        print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

        # Feature selection
        print('Running Feature Selector')
        fs = utils.FeatureSelector()
        X_train_embedded = fs.fit_transform(X_train)
        X_test_embedded = fs.transform(X_test)
        print(f"Feature dimensions after selection: {X_train_embedded.shape[1]}")

        # Train model
        print('Building Model')
        classifier = utils.Classifier()
        classifier.fit(X_train_embedded, y_train)
        print(f"Classifier parameters: {classifier.clf.best_params_}")

        results = []
        for k, X, y in zip(['train', 'test'], [X_train_embedded, X_test_embedded], [y_train, y_test]):
            print(f"Running Prediction for {k} Set")
            y_predicted = classifier.predict(X)
            accuracy = accuracy_score(y, y_predicted)
            precision = precision_score(y, y_predicted)
            recall = recall_score(y, y_predicted)

            pred_df = pd.DataFrame([list(y.values), list(y_predicted)]).T
            pred_df.columns = ['actual', 'predicted']
            one_classified_as_two = pred_df[np.logical_and(pred_df.actual == 1,
                                        pred_df.predicted == 2)].shape[0] / pred_df[pred_df.actual == 1].shape[0]
            two_classified_as_one = pred_df[np.logical_and(pred_df.actual == 2,
                                        pred_df.predicted == 1)].shape[0] / pred_df[pred_df.actual == 2].shape[0]
            print(f"\t1->2 {one_classified_as_two} | 2->1 {two_classified_as_one}")

            results.append((k, accuracy, precision, recall))

        return pd.DataFrame(
            results, columns=['data', 'accuracy', 'precision', 'recall']
        )


if __name__ == '__main__':
    # Ensure a command-line argument is provided (input CSV file path)
    assert len(sys.argv) > 1
    
    # Initialize the Evaluator with the input CSV file path
    evaluator = Evaluator(sys.argv[1])
    
    # Perform evaluation and print the results
    evaluation_results = evaluator.evaluate()
    print(evaluation_results)
