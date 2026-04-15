"""
Configuration file for NYC Taxi Trip Duration project.
Centralized settings for paths, model parameters, and data filtering.
"""
import os
from pathlib import Path

# path to dataset  
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "input"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file paths (update these for your environment)
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
TEST_FILE = DATA_DIR / "test.csv"

# Model output paths
MODEL_SAVE_DIR = MODELS_DIR
PIPELINE_FILENAME = "ridge_pipeline.pkl"

# hyperparameters 

# Ridge Regression
RIDGE_PARAMS = {
    "alpha": 1.0,
    "degree": 2  # Polynomial degree
}


TRAIN_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "shuffle": True
}



def get_model_path(filename: str = PIPELINE_FILENAME) -> Path:
    """Get full path for model file."""
    return MODEL_SAVE_DIR / filename


def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
