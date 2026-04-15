<p align="center">
  <h1 align="center">🚕 NYC Taxi Trip Duration Prediction</h1>
  <p align="center">
    <strong>End-to-end machine learning pipeline for predicting taxi trip durations in New York City</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-results">Results</a> •
    <a href="#-usage">Usage</a> •
    <a href="#-contributing">Contributing</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/pandas-1.3%2B-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
    <img src="https://img.shields.io/badge/code%20style-PEP8-000000?style=flat-square" alt="Code Style: PEP8">
  </p>
</p>

---

## Overview

A production-ready regression pipeline that predicts NYC taxi trip durations using geospatial feature engineering and polynomial Ridge regression. Built on the [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) dataset, this project demonstrates a clean, reproducible ML workflow — from raw GPS coordinates to calibrated duration estimates.

**Key results:** Achieves **R² = 0.68** and **95.14% MAPE accuracy** on the held-out test set with a lightweight, interpretable model.

---

## Features

| Category | Details |
|:---|:---|
| **Feature Engineering** | Haversine & Manhattan distances, bearing, temporal decomposition (hour, day-of-week, month), binary indicators (weekend, night, peak hour) |
| **Preprocessing** | Log-transform target for skew correction, IQR-based outlier removal, polynomial feature expansion (degree 2) |
| **Model** | `Ridge` regression with `StandardScaler` inside a serializable `sklearn.Pipeline` |
| **Inference** | Single-command batch prediction with automatic inverse transform (`expm1`) |
| **Exploration** | Full EDA notebook with distribution plots, correlation heatmaps, outlier detection, and time-series analysis |
| **Logging** | Structured Python logging with configurable levels and formatters |

---
<!-- 
## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Raw CSV Data                          │
│          (pickup/dropoff GPS, datetime, metadata)            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│               Feature Engineering Pipeline                    │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │  Temporal   │ │  Geospatial  │ │   Target Transform      │ │
│  │  Extraction │ │  Distances   │ │   log1p(trip_duration)  │ │
│  │            │ │              │ │                         │ │
│  │ • hour     │ │ • haversine  │ │   Outlier Removal       │ │
│  │ • month    │ │ • manhattan  │ │   IQR-based filtering   │ │
│  │ • dow      │ │ • bearing    │ │                         │ │
│  │ • weekend  │ │              │ │                         │ │
│  │ • night    │ │              │ │                         │ │
│  │ • peak_hr  │ │              │ │                         │ │
│  └────────────┘ └──────────────┘ └─────────────────────────┘ │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│               sklearn Pipeline (serialized)                   │
│                                                              │
│   PolynomialFeatures(degree=2) → StandardScaler() → Ridge() │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│          Predictions (expm1 inverse transform)               │
│              trip_duration in seconds                         │
└──────────────────────────────────────────────────────────────┘
```
 -->
---

## Results

### Model Performance

| Split | R² Score | MAE | RMSE | MAPE Accuracy |
|:------|:--------:|:---:|:----:|:-------------:|
| **Train** | 0.6800 | 0.3041 | 0.4020 | 95.16% |
| **Validation** | 0.6809 | 0.3039 | 0.4028 | 95.17% |
| **Test** | 0.6789 | 0.3056 | 0.4039 | 95.14% |

> **Note:** Metrics are computed on log-transformed targets (`log1p`). The near-identical train/val/test performance indicates the model generalizes well with minimal overfitting.

### Engineered Features

| Feature | Type | Description |
|:--------|:-----|:------------|
| `haversine` | Continuous | Great-circle distance between pickup and dropoff (log1p-transformed) |
| `manhattan` | Continuous | L1 distance between coordinates |
| `bearing` | Continuous | Initial compass heading from pickup to dropoff |
| `hour` | Discrete | Hour of pickup (0–23) |
| `dayofweek` | Discrete | Day of week (0=Mon, 6=Sun) |
| `month` | Discrete | Month of pickup |
| `dayofyear` | Discrete | Day of year (1–366) |
| `is_weekend` | Binary | Saturday or Sunday |
| `is_night` | Binary | Pickup between 00:00–05:59 |
| `is_peak_hour` | Binary | Rush hour (7–9 AM, 4–6 PM) |
| `passenger_count` | Discrete | Number of passengers |
| `store_and_fwd_flag` | Binary | Whether trip data was held in vehicle memory |

---

## Repository Structure

```
NYC-Taxi-Trip-Duration/
├── src/
│   ├── config.py            # Centralized paths, hyperparameters, and project settings
│   ├── data_helper.py       # Data loading, feature engineering, and preprocessing
│   ├── data_staticts.py     # EDA utilities: plots, statistics, outlier detection
│   ├── inference.py         # CLI batch inference with trained pipeline
│   ├── logger.py            # Structured logging configuration
│   ├── train.py             # Training loop with train/val/test evaluation
│   └── utils.py             # Metric helpers (R², MAE, RMSE, MAPE accuracy)
├── models/                  # Saved model artifacts (.pkl)
├── notebooks/
│   └── NYC Taxi Trip Duration(EDA).ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Boules123/NYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration
```

### 2. Set Up a Virtual Environment

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) CSV in the data directory:

```
data/nyc_taxi_trip_duration.csv
```

> The dataset should contain these columns: `id`, `pickup_datetime`, `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `passenger_count`, `store_and_fwd_flag`, `trip_duration`.

---

## Usage

### Training

Run the training pipeline from the project root:

```bash
python -m src.train
```

**What happens:**

1. Loads and splits data into train (80%) / validation (10%) / test (10%)
2. Applies the full feature engineering pipeline (temporal, geospatial, outlier removal)
3. Fits a `PolynomialFeatures → StandardScaler → Ridge` pipeline
4. Evaluates and prints R², MAE, RMSE, and MAPE accuracy for all splits
5. Serializes the trained pipeline to `ridge_pipeline_r2_<score>.pkl`

### Inference

Run batch predictions on new data:

```bash
python -m src.inference \
  --test path/to/test.csv \
  --pipeline path/to/ridge_pipeline_r2_0.68.pkl \
  --output predictions.csv
```

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--test` | `test.csv` | Path to the input CSV file |
| `--pipeline` | `pipeline.pkl` | Path to the serialized model pipeline |
| `--output` | `None` | *(Optional)* Path to save predictions CSV |

**Output schema:**

| Column | Description |
|:-------|:------------|
| `id` | Original sample identifier |
| `trip_duration` | Predicted duration in seconds |

> Predictions are automatically inverse-transformed from log-space back to seconds via `expm1`.

### Exploratory Data Analysis

Launch the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/NYC\ Taxi\ Trip\ Duration\(EDA\).ipynb
```

The EDA toolkit (`src/data_staticts.py`) provides:
- `describe_data()` — Shape, dtypes, missing values, five-number summary
- `plot_distribution()` — Histograms for any numerical column
- `plot_correlation_matrix()` — Annotated correlation heatmap
- `plot_boxplot()` — Outlier visualization
- `plot_scatter()` — Bivariate scatter plots
- `plot_time_series()` — Temporal trend plots
- `detect_outliers_iqr()` — Statistical outlier detection with IQR bounds

---

## Configuration

All hyperparameters and project paths are centralized in [`src/config.py`](src/config.py):

```python
# Hyperparameters
RIDGE_PARAMS = {
    "alpha": 1.0,       # L2 regularization strength
    "degree": 2          # Polynomial feature expansion degree
}

# Train/validation/test split ratios
TRAIN_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "cv_folds": 5,
    "shuffle": True
}
```

---

## Tech Stack

| Library | Version | Purpose |
|:--------|:--------|:--------|
| [NumPy](https://numpy.org/) | ≥ 1.21.0 | Numerical computing |
| [pandas](https://pandas.pydata.org/) | ≥ 1.3.0 | Data manipulation & I/O |
| [scikit-learn](https://scikit-learn.org/) | ≥ 1.0.0 | ML pipeline, model, metrics |
| [Matplotlib](https://matplotlib.org/) | ≥ 3.4.0 | Static visualizations |
| [Seaborn](https://seaborn.pydata.org/) | ≥ 0.11.0 | Statistical plots |
| [joblib](https://joblib.readthedocs.io/) | ≥ 1.1.0 | Model serialization |
| [SciPy](https://scipy.org/) | ≥ 1.7.0 | Scientific computing utilities |

---

## Roadmap

- [ ] Add CLI argument for custom dataset path and model output directory
- [ ] Save trained artifacts to `models/` by default
- [ ] Add cross-validation with hyperparameter grid search
- [ ] Integrate experiment tracking (MLflow / Weights & Biases)
- [ ] Add unit and integration tests (`pytest`)
- [ ] Implement gradient-boosted model baseline (XGBoost / LightGBM) for comparison
- [ ] Add Docker support for reproducible environments
- [ ] Deploy as REST API with FastAPI

---

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "feat: add my feature"`
4. **Push** to your branch: `git push origin feature/my-feature`
5. Open a **Pull Request**

Please ensure your code follows PEP 8 conventions and includes appropriate docstrings.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
