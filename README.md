# NYC Taxi Trip Duration Prediction

A machine learning project to predict NYC taxi trip duration using geographic and temporal features.

## Project Structure

```
project_1/
├── src/
│   ├── config.py         # Centralized configuration settings
│   ├── data_helper.py    # Data loading and feature engineering
│   ├── data_staticts.py  # Statistical analysis and visualization
│   ├── train.py          # Model training pipeline
│   ├── inference.py      # Prediction on new data
│   └── utils.py          # Utility functions and metrics
├── input/                # Data files (train.csv, val.csv, test.csv)
├── models/               # Saved model pipelines
├── notebooks/            # Jupyter notebooks for EDA
└── requirements.txt      # Python dependencies
```

## Features

### Data Quality Filters
- **Geographic Boundaries**: Filters trips outside NYC bounds
- **Coordinate Validation**: Removes invalid (0 or NaN) coordinates  
- **Trip Duration Constraints**: Filters unrealistic durations (< 1 min or > 3 hours)
- **Passenger Count Validation**: Ensures valid passenger counts (1-6)
- **Distance Filtering**: Removes trips with unrealistic distances

### Feature Engineering
- **Temporal Features**: Hour, day of week, month, day of year
- **Binary Indicators**: Weekend, night, peak hour
- **Distance Features**: Haversine, Manhattan distance, bearing
- **Location Features**: Airport proximity, Manhattan zone detection
- **Target Transform**: Log transformation for trip duration

### Airports Detected
- JFK International Airport
- LaGuardia Airport  
- Newark Liberty International (EWR)

## Configuration

All settings are centralized in `src/config.py`:

```python
# NYC Geographic Boundaries
NYC_BOUNDS = {
    "min_lat": 40.4774,
    "max_lat": 40.9176,
    "min_lon": -74.2591,
    "max_lon": -73.7004
}

# Filter Thresholds
MIN_TRIP_DURATION = 60      # 1 minute
MAX_TRIP_DURATION = 10800   # 3 hours
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6

# Model Hyperparameters
RIDGE_PARAMS = {"alpha": 1.0, "degree": 2}
```

## Usage

### Training
```bash
cd src
python train.py
```

### Inference
```bash
cd src
python inference.py --test path/to/test.csv --output predictions.csv
```

### Python API
```python
from data_helper import load_data, prepare_data_pipeline, drop_coordinate_columns
from train import train_model

# Load and prepare data
df = load_data("train.csv")
df = prepare_data_pipeline(df, is_train=True, apply_filters=True)
df = drop_coordinate_columns(df)

# Split features and target
X = df.drop('trip_duration', axis=1)
y = df['trip_duration']
```

## Model Pipeline

1. **PolynomialFeatures**: Generates polynomial and interaction features
2. **StandardScaler**: Normalizes features (zero mean, unit variance)
3. **Ridge Regression**: L2 regularized linear regression

## Metrics

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE Accuracy**: Mean Absolute Percentage Error based accuracy

## Data Requirements

Input CSV files must include:
- `id`: Unique trip identifier
- `pickup_datetime`: Date and time of pickup
- `pickup_latitude`, `pickup_longitude`: Pickup coordinates
- `dropoff_latitude`, `dropoff_longitude`: Dropoff coordinates
- `passenger_count`: Number of passengers
- `store_and_fwd_flag`: Y/N flag for store and forward
- `trip_duration`: Target variable (training only)

## Logging

The project uses Python's logging module. Logs include:
- Data loading and filtering statistics
- Pipeline processing steps
- Model training metrics
- Inference results

To adjust log level, modify `LOG_LEVEL` in `config.py`.
