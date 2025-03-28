"""Module for model inference functions."""
import os
from typing import Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .preprocess import clean_data, encode_features, scale_features


ModelObjects = Tuple[
    RandomForestRegressor,
    StandardScaler,
    OneHotEncoder,
]


def load_objects(models_dir: str = None) -> ModelObjects:
    """Load model and preprocessing objects.

    Args:
        models_dir: Directory containing saved objects

    Returns:
        Tuple of (model, scaler, encoder)
    """
    if models_dir is None:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')

    model = joblib.load(os.path.join(models_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    encoder = joblib.load(os.path.join(models_dir, 'encoder.joblib'))
    return model, scaler, encoder


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions on new data.

    Args:
        input_data: DataFrame containing features to predict on

    Returns:
        Array of predictions
    """
    # Load model objects
    model, scaler, encoder = load_objects()

    # Clean data
    cleaned_data = clean_data(input_data)

    # Get feature lists
    cat_cols = cleaned_data.select_dtypes(include=['object']).columns.tolist()
    num_cols = cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical features
    encoded_data, _ = encode_features(
        cleaned_data, cat_cols, fit=False, encoder=encoder
    )

    # Scale numeric features
    processed_data, _ = scale_features(
        encoded_data, num_cols, fit=False, scaler=scaler
    )

    # Make predictions
    return model.predict(processed_data)
