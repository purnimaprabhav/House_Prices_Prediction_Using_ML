"""
This module contains functions for model training and evaluation.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, r2_score
import joblib

from .preprocess import initialize_transformers, preprocess_data


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Build and train the model, including all preprocessing steps.

    Args:
        data: Input DataFrame containing features and target

    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Define features
    numerical_features = ['GrLivArea', 'TotalBsmtSF']
    categorical_features = ['Neighborhood', 'ExterQual']
    target = 'SalePrice'

    # Split data
    X = data[numerical_features + categorical_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize transformers
    transformers = initialize_transformers()

    # Preprocess training data
    X_train_processed = preprocess_data(
        X_train,
        numerical_features,
        categorical_features,
        transformers,
        fit=True
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Save model and transformers
    joblib.dump(model, '../models/model.joblib')
    for name, transformer in transformers.items():
        joblib.dump(transformer, f'../models/{name}.joblib')

    # Preprocess test data
    X_test_processed = preprocess_data(
        X_test,
        numerical_features,
        categorical_features,
        transformers,
        fit=False
    )

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test_processed)
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'rmsle': round(rmsle, 4),
        'r2': round(r2, 4)
    }
