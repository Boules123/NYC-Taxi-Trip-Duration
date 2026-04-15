"""
Inference module for NYC Taxi Trip Duration prediction.
Loads trained pipeline and makes predictions on new data.
"""
import os
import numpy as np
import pandas as pd
import joblib

from .data_helper import load_data, prepare_data_pipeline, drop_coordinate_columns


def load_pipeline(pipeline_path='pipeline.pkl'):
    """
    load trained model.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    return joblib.load(pipeline_path)


def predict(pipeline, data):
    predictions = pipeline.predict(data)
    # Reverse log1p transformation applied during training
    predictions = np.expm1(predictions)
    return predictions


def run_inference(test_file, pipeline_path='pipeline.pkl', save_predictions=None):
    """
    Run complete inference pipeline on test data.
    """    
    # Load and preprocess data
    df = load_data(test_file)
    original_len = len(df)
    print(f"   Loaded {original_len} samples")
    
    # Store IDs if present for output
    ids = df['id'].values if 'id' in df.columns else None
    
    df = prepare_data_pipeline(df, is_train=False)
    df = drop_coordinate_columns(df)
    print(f"   Features: {list(df.columns)}")
    
    # Load pipeline and predict
    pipeline = load_pipeline(pipeline_path)
    print(f"   Loaded: {pipeline_path}")
    
    predictions = predict(pipeline, df)
    print(f"Generated {len(predictions)} predictions")
    print(f"Duration range: {predictions.min():.0f}s - {predictions.max():.0f}s")
    print(f"Mean duration: {predictions.mean():.0f}s ({predictions.mean()/60:.1f} min)")
    
    # Optionally save predictions
    if save_predictions:
        print(f"\n5. Saving predictions to {save_predictions}...")
        output_df = pd.DataFrame({
            'id': ids if ids is not None else range(len(predictions)),
            'trip_duration': predictions
        })
        output_df.to_csv(save_predictions, index=False)
        print(f"   Saved {len(output_df)} predictions")

    return predictions


def main():
    """Main inference entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NYC Taxi Trip Duration Inference')
    parser.add_argument('--test', default='test.csv', help='Path to test CSV')
    parser.add_argument('--pipeline', default='pipeline.pkl', help='Path to pipeline file')
    parser.add_argument('--output', default=None, help='Path to save predictions CSV')
    
    args = parser.parse_args()
    
    predictions = run_inference(
        test_file=args.test,
        pipeline_path=args.pipeline,
        save_predictions=args.output
    )
    
    return predictions


if __name__ == '__main__':
    main()

