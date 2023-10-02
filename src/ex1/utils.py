from itertools import combinations

import pandas as pd
from dtaidistance import dtw
from scipy.signal import detrend


def pre_process(data: dict) -> pd.DataFrame:
    """
    Pre-processes the input data dictionary and returns a pre-processed DataFrame.

    Args:
        data (dict): A dictionary containing device data as values.

    Returns:
        data_df (DataFrame): A pre-processed DataFrame with scaled, detrended, and smoothed data.
    """
    # @TODO: this code can be run in parallel as each series is processed independently of others
    for v in data.values():
        # Scale the data
        v.value = (v.value - v.value.mean()) / v.value.std()
        # Detrend the data
        v.value = detrend(v.value)
        # Smooth the data
        # @TODO: we can use more sophisticated smoothing methods here
        v.value = v.value.rolling(5).mean()

    # Concatenate data values into a DataFrame
    data_df = pd.concat(data.values(), axis=1)
    data_df.columns = data.keys()

    # Add frequency-related columns based on datetime
    data_df['freq_5min'] = data_df.reset_index()['datetime'].apply(lambda x: x.to_period('5min')).values
    data_df['freq_hour'] = data_df.reset_index()['datetime'].apply(lambda x: x.to_period('H')).values
    data_df['freq_date'] = data_df.reset_index()['datetime'].apply(lambda x: x.to_period('D')).values
    data_df['freq_week'] = data_df.reset_index()['datetime'].apply(lambda x: x.to_period('W')).values
    data_df['freq_month'] = data_df.reset_index()['datetime'].apply(lambda x: x.to_period('M')).values

    return data_df

def get_distances(data_df: pd.DataFrame, keys: set) -> pd.DataFrame:
    """
    Calculates and returns DTW distances between pairs of devices based on median values
    per week.

    Args:
        data_df (DataFrame): Pre-processed data DataFrame.
        keys (set): Set of device names.

    Returns:
        distances_df (DataFrame): DataFrame with device pairs and their DTW distances.
    """
    distances = []
    # @TODO: this code can be made parallel as each pairwise timeseries is processed independently
    for device1, device2 in list(combinations(keys, 2)):
        _df = data_df.groupby('freq_week').median()[[device1, device2]].dropna()
        distance = dtw.distance_fast(_df[device1].values, _df[device2].values, use_pruning=True)
        distances.append((device1, device2, distance))
    return pd.DataFrame(distances, columns=['device1', 'device2', 'dist'])

def get_correlations(distances_df: pd.DataFrame) -> pd.Series:
    """
    Calculates descriptive correlations based on DTW distances.

    Args:
        distances_df (DataFrame): DataFrame containing device pairs and their DTW distances.

    Returns:
        correlations (Series): Series with descriptive correlations for each device pair.
    """
    dist_stats = distances_df.dist.describe()
    
    def get_descriptive_correlation(x):
        if x <= dist_stats['25%']:
            return 'correlated'
        elif x <= dist_stats['50%']:
            return 'loose correlation'
        else:
            return 'uncorrelated'
    
    # Apply the function to calculate descriptive correlations
    return distances_df.dist.apply(lambda x: get_descriptive_correlation(x))
