import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def describe_data(df):
    """Print basic statistics about the dataframe."""
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())


def plot_distribution(df, column, bins=50, figsize=(10, 6)):
    """Plot histogram distribution of a numerical column."""
    plt.figure(figsize=figsize)
    plt.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """Plot correlation heatmap for numerical columns."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_boxplot(df, column, figsize=(10, 6)):
    """Plot boxplot to visualize outliers."""
    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column])
    plt.xlabel(column)
    plt.title(f'Boxplot of {column}')
    plt.show()


def plot_scatter(df, x_col, y_col, figsize=(10, 6)):
    """Plot scatter plot between two columns."""
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_time_series(df, time_col, value_col, figsize=(14, 6)):
    """Plot time series data."""
    plt.figure(figsize=figsize)
    df_sorted = df.sort_values(time_col)
    plt.plot(df_sorted[time_col], df_sorted[value_col], alpha=0.7)
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.title(f'{value_col} over {time_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_categorical_counts(df, column, figsize=(10, 6)):
    """Plot bar chart for categorical column counts."""
    plt.figure(figsize=figsize)
    df[column].value_counts().plot(kind='bar', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Counts of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    return outliers


def summary_statistics(df, column):
    """Print detailed statistics for a single column."""
    stats = {
        'Count': df[column].count(),
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Std': df[column].std(),
        'Min': df[column].min(),
        'Max': df[column].max(),
        'Q1': df[column].quantile(0.25),
        'Q3': df[column].quantile(0.75),
        'Skewness': df[column].skew(),
        'Kurtosis': df[column].kurtosis()
    }
    
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    return stats

def data_statictis_pipeline(df):
    """Create a pipeline for data preprocessing."""
    describe_data(df)
    plot_distribution(df, 'trip_duration')
    plot_correlation_matrix(df)
    plot_boxplot(df, 'trip_duration')
    plot_scatter(df, 'trip_duration', 'trip_duration')
    plot_time_series(df, 'pickup_datetime', 'trip_duration')
    plot_categorical_counts(df, 'vendor_id')
    detect_outliers_iqr(df, 'trip_duration')
    summary_statistics(df, 'trip_duration')