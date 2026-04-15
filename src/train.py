"""
Training module for NYC Taxi Trip Duration prediction.
Uses Polynomial Ridge regression with proper validation and metrics.
"""
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from .data_helper import load_data, prepare_data_pipeline, drop_coordinate_columns
from .utils import calculate_r2, calculate_accuracy
from .logger import setup_logging

def create_pipeline(alpha=1.0, degree=2):
    """
    A scikit-learn Pipeline with polynomial features and Ridge regression.
    """
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    return pipeline


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Evaluate model and return metrics dictionary."""
    r2 = calculate_r2(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_acc = calculate_accuracy(y_true, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE Accuracy: {mape_acc:.2f}%")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape_acc': mape_acc
    }


def train_model(x_train, y_train, x_val, y_val, alpha=1.0):
    """
    Train pipeline and evaluate on both train and validation sets.
    x_train, y_train: Training features and target
    x_val, y_val: Validation features and target
    alpha: Regularization strength for Ridge regression
    """
    pipeline = create_pipeline(alpha=alpha)
    pipeline.fit(x_train, y_train)
    
    # Evaluate on training set
    y_train_pred = pipeline.predict(x_train)
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    
    # Evaluate on validation set
    y_val_pred = pipeline.predict(x_val)
    val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
    
    return pipeline, {'train': train_metrics, 'val': val_metrics}


def main():
    """Main training pipeline."""
    logger = setup_logging()
    
    # Load and preprocess data
    df_train, df_val, df_test = load_data('data/nyc_taxi_trip_duration.csv', split=True)
    
    logger.info(f"Loaded train.csv with {len(df_train)} samples")
    logger.info(f"Loaded val.csv with {len(df_val)} samples")
    logger.info(f"Loaded test.csv with {len(df_test)} samples")
    
    # ensure_directories()

    # Feature engineering and drop coordinate columns for all sets
    df_train = prepare_data_pipeline(df_train)
    df_train = drop_coordinate_columns(df_train)
    
    df_val = prepare_data_pipeline(df_val)
    df_val = drop_coordinate_columns(df_val) 
    
    df_test = prepare_data_pipeline(df_test) 
    df_test = drop_coordinate_columns(df_test)
    
    # Prepare features and targets
    x_train = df_train.drop('trip_duration', axis=1)
    y_train = df_train['trip_duration']
    x_val = df_val.drop('trip_duration', axis=1)
    y_val = df_val['trip_duration']
    x_test = df_test.drop('trip_duration', axis=1)
    y_test = df_test['trip_duration']

    # Train and evaluate pipeline
    pipeline, metrics = train_model(x_train, y_train, x_val, y_val)
    
    # Save pipeline (includes scaler + model)
    pipeline_filename = f'ridge_pipeline_r2_{metrics["val"]["r2"]:.2f}.pkl'
    joblib.dump(pipeline, pipeline_filename)
    logger.info(f"Pipeline saved as: {pipeline_filename}")

    # Test pipeline on test data
    y_test_pred = pipeline.predict(x_test)
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

if __name__ == '__main__':
    main()

# Training and evaluation results:
# Training Metrics:
#   R2 Score: 0.6800
#   MAE: 0.3041
#   RMSE: 0.4020
#   MAPE Accuracy: 95.16%
#   Pipeline saved as: ridge_pipeline_r2_0.68.pkl

# Validation Metrics:
#   R2 Score: 0.6809
#   MAE: 0.3039
#   RMSE: 0.4028
#   MAPE Accuracy: 95.17%

# Test Metrics:
#   R2 Score: 0.6789
#   MAE: 0.3056
#   RMSE: 0.4039
#   MAPE Accuracy: 95.14%
