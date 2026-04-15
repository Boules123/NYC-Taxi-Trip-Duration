"""
Utility classes and functions for data preprocessing and evaluation.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Standalone helper functions for use without class instance
def calculate_accuracy(y_true, y_pred):
    """
    Calculate MAPE-based accuracy for regression.
    Returns 100 - MAPE as an accuracy percentage.
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return max(0, 100 - mape)


def calculate_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))
