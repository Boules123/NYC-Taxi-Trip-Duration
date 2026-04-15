# NYC Taxi Trip Duration Prediction

Predict NYC taxi trip duration from pickup/dropoff coordinates and pickup time using a feature-engineered Ridge Regression pipeline.

This repository focuses on a clean, educational baseline with practical preprocessing, distance-based features, and a reproducible training + inference flow.

## Highlights

- End-to-end regression pipeline with `scikit-learn`
- Feature engineering for temporal and geospatial signals
- Log-transform strategy for skewed duration targets
- Outlier handling for more stable model fit
- Simple CLI for training and batch inference
- EDA notebook included for exploration

## Model Overview

The training pipeline is:

1. `PolynomialFeatures(degree=2, include_bias=False)`
2. `StandardScaler()`
3. `Ridge(alpha=1.0)`

Primary metrics reported:

- R2 Score
- MAE
- RMSE
- MAPE-based accuracy (`100 - MAPE`)

## Repository Structure

```text
project_1/
|- src/
|  |- config.py
|  |- data_helper.py
|  |- data_staticts.py
|  |- inference.py
|  |- logger.py
|  |- train.py
|  |- utils.py
|- models/
|- notebooks/
|  |- NYC Taxi Trip Duration(EDA).ipynb
|- requirements.txt
|- LICENSE
```

## Quick Start

### 1) Clone

```bash
git clone https://github.com/Boules123/NYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration
```

### 2) Create environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add dataset

Current training code expects one file at:

```text
data/nyc_taxi_trip_duration.csv
```

Create the `data/` folder in the project root and place your CSV there.

## Data Schema

Required columns for training:

- `id`
- `pickup_datetime`
- `pickup_latitude`
- `pickup_longitude`
- `dropoff_latitude`
- `dropoff_longitude`
- `passenger_count`
- `store_and_fwd_flag` (Y/N)
- `trip_duration` (target, in seconds)

Optional columns supported by preprocessing (dropped if present):

- `dropoff_datetime`
- `vendor_id`

## Training

Run from project root:

```bash
python -m src.train
```

What happens during training:

- Loads `data/nyc_taxi_trip_duration.csv`
- Splits into train/val/test
- Applies preprocessing and feature engineering
- Trains Ridge pipeline
- Prints metrics for train/validation/test
- Saves model as `ridge_pipeline_r2_<score>.pkl` in the current working directory

## Inference

Run from project root:

```bash
python -m src.inference --test path/to/test.csv --pipeline path/to/ridge_pipeline_r2_xx.pkl --output predictions.csv
```

Output format:

- `id`
- `trip_duration` (predicted seconds)

Notes:

- Inference applies the same feature engineering as training.
- Predictions are transformed back to seconds with `expm1`.

## Feature Engineering Details

- Datetime features: `dayofweek`, `month`, `hour`, `dayofyear`
- Binary indicators: `is_weekend`, `is_night`, `is_peak_hour`
- Distance features:
    - `haversine` (then `log1p`)
    - `manhattan`
    - `bearing`
- Target transform: `trip_duration = log1p(trip_duration)` during training
- Outlier removal: IQR-based filtering on transformed target

## Exploratory Data Analysis

Use the notebook and helper module:

- `notebooks/NYC Taxi Trip Duration(EDA).ipynb`
- `src/data_staticts.py`

The EDA utilities include:

- Distribution plots
- Correlation heatmaps
- Outlier inspection
- Summary statistics

## Configuration

Project constants are centralized in `src/config.py` (paths, model params, training config).

Current default model settings:

```python
RIDGE_PARAMS = {
        "alpha": 1.0,
        "degree": 2
}
```

## Known Limitations

- Training path is currently hardcoded to `data/nyc_taxi_trip_duration.csv`.
- Trained model is saved to current working directory, not automatically under `models/`.
- No automated test suite yet.

## Suggested Next Improvements

- Add argparse options for train dataset path and model output path
- Save artifacts under `models/` by default
- Add unit tests for preprocessing and metrics
- Add cross-validation and experiment tracking

## License

This project is licensed under the MIT License. See `LICENSE`.
