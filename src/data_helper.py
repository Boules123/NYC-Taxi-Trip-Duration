"""
Data loading and feature engineering functions for NYC Taxi Trip Duration.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, split=False):
    """Load CSV data from file path."""
    if split:
        df = pd.read_csv(file_path)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        return train_df, val_df, test_df
    return pd.read_csv(file_path)


def feature_engineering(data, is_train=True):
    """
    Perform feature engineering on the dataset.
    - Extracts datetime features
    - Encodes categorical variables
    - Applies log transformation to target
    """
    # Drop ID column
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    # Extract datetime features
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dayofweek'] = data.pickup_datetime.dt.dayofweek
    data['month'] = data.pickup_datetime.dt.month
    data['hour'] = data.pickup_datetime.dt.hour
    data['dayofyear'] = data.pickup_datetime.dt.dayofyear

    # Encode store_and_fwd_flag
    data['store_and_fwd_flag'] = data.store_and_fwd_flag.map({'N': 0, 'Y': 1})

    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int) # 5,6 are weekend
    data['is_night'] = data['hour'].between(0, 5).astype(int) # 0,1,2,3,4,5 are night
    data['is_peak_hour'] = data['hour'].isin([7,8,9,16,17,18]).astype(int) # 7,8,9,16,17,18 are peak hours
    
    # Drop unnecessary columns
    data.drop(columns=['pickup_datetime'], inplace=True)
    if 'dropoff_datetime' in data.columns:
        data.drop(columns=['dropoff_datetime'], inplace=True)
    if 'vendor_id' in data.columns:
        data.drop(columns=['vendor_id'], inplace=True)
    
    # Log transform target for training
    if is_train and 'trip_duration' in data.columns:
        data['trip_duration'] = np.log1p(data.trip_duration)
    
    return data


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points in miles.
    """
    R = 3958.8  # Earth radius in miles
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_haversine_feature(df):
    """Add haversine distance feature between pickup and dropoff."""
    df['haversine'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    return df


def add_manhattan_feature(df):
    """Add Manhattan distance feature between pickup and dropoff."""
    df['manhattan'] = (
        np.abs(df['pickup_latitude'] - df['dropoff_latitude']) +
        np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
    )
    return df


def add_bearing_feature(df):
    """Add bearing (direction) feature between pickup and dropoff."""
    lat1 = np.radians(df['pickup_latitude'])
    lon1 = np.radians(df['pickup_longitude'])
    lat2 = np.radians(df['dropoff_latitude'])
    lon2 = np.radians(df['dropoff_longitude'])
    
    dlon = lon2 - lon1
    x = np.cos(lat2) * np.sin(dlon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    df['bearing'] = np.arctan2(x, y)
    return df

# remove outliers from trip_duration
def remove_outliers(df, column='trip_duration'):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


def drop_coordinate_columns(df):
    """Drop raw latitude/longitude columns after distance features are extracted."""
    cols_to_drop = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    cols_present = [col for col in cols_to_drop if col in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df


def prepare_data_pipeline(df, is_train=True):
    """
    Complete data preparation pipeline.
    Applies feature engineering and distance features.
    """
    df = feature_engineering(df, is_train=is_train)
    df = add_haversine_feature(df)
    df['haversine'] = np.log1p(df['haversine'])
    df = add_manhattan_feature(df)
    df = add_bearing_feature(df)
    if is_train:
        df = remove_outliers(df)
    return df