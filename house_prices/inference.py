"""
This module contains functions for model inference.
"""

from typing import Dict
import pandas as pd
import numpy as np
import joblib

from .preprocess import preprocess_data


def load_transformers() -> Dict:
    """
    Load all transformers from disk.

    Returns:
        dict: Dictionary containing all loaded transformers
    """
    transformers = {}
    transformer_names = ['num_imputer', 'cat_imputer', 'scaler', 'encoder']
    
    for name in transformer_names:
        transformers[name] = joblib.load(f'../models/{name}.joblib')
    
    return transformers


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data using the trained model.

    Args:
        input_data: DataFrame containing features to predict on

    Returns:
        numpy.ndarray: Array of predictions
    """
    # Define features
    numerical_features = ['GrLivArea', 'TotalBsmtSF']
    categorical_features = ['Neighborhood', 'ExterQual']

    # Load model and transformers
    model = joblib.load('../models/model.joblib')
    transformers = load_transformers()

    # Preprocess input data
    X_processed = preprocess_data(
        input_data,
        numerical_features,
        categorical_features,
        transformers,
        fit=False
    )

    # Make predictions
    return model.predict(X_processed)
