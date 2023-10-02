import pickle
import sys
from warnings import simplefilter

import pandas as pd
from dtaidistance import dtw
from utils import get_correlations, get_distances, pre_process

# @TODO: all prints should be handled through logging. this will allows us to
# build process monitoring infrastructure in the future
class CorrelationsGenerator:
    """
    A class for generating correlations between time series data of different devices.

    Methods:
        get_correlations(data: dict) -> tuple:
            Preprocesses the data, calculates distances, and computes correlations.

    Usage:
        correlation_generator = CorrelationsGenerator()
        data_df, device_correlations_df = correlation_generator.get_correlations(data)
    """

    def __init__(self):
        """
        Initializes the CorrelationsGenerator instance.
        """
        simplefilter(action='ignore', category=FutureWarning)

    def get_correlations(self, data: dict) -> tuple:
        """
        Preprocesses data, calculates distances, and computes correlations.

        Args:
            data (dict): A dictionary containing time series data for different devices.

        Returns:
            tuple: A tuple containing preprocessed data DataFrame and device correlations DataFrame.
        """
        # Preprocess data
        print('Running preprocessing')
        data_df = pre_process(data)
        print(f"Shape of preprocessed data {data_df.shape}")

        # Calculate distances
        print('Calculating distances')
        distances = get_distances(data_df, data.keys())
        distances_df = pd.DataFrame(distances, columns=['device1', 'device2', 'dist'])
        print(f"Descriptive stats of distances {distances_df.describe()}")

        # Calculate correlations
        print('Calculating correlations')
        correlations = get_correlations(distances_df)
        distances_df['correlation'] = correlations

        return data_df, distances_df.sort_values(by=['correlation', 'device1', 'device2'])

if __name__ == '__main__':
    assert len(sys.argv) > 1

    # Load data from a pickle file
    file = open(sys.argv[1], 'rb')
    data = pickle.load(file)

    # Initialize CorrelationsGenerator
    correlation_generator = CorrelationsGenerator()

    # Generate correlations and get results
    _, device_correlations_df = correlation_generator.get_correlations(data)
    print(device_correlations_df)
