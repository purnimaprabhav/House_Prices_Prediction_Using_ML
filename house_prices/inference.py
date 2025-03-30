<<<<<<< HEAD
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
=======
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from house_prices.preprocess import preprocess_features

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message=(
        "The number of features does not match the model's expected input!"
    ),
)

# Add custom module path
sys.path.append("/Users/purnimaprabha/dsp-purnima-prabha")

# Load models and transformers
model = joblib.load(
    "/Users/purnimaprabha/dsp-purnima-prabha/models/random_forest_model.pkl"
)
encoder = joblib.load("models/one_hot_encoder.pkl")
scaler = joblib.load("models/standard_scaler.pkl")
cat_imputer = joblib.load("models/cat_imputer.pkl")


def make_predictions(input_data: pd.DataFrame) -> pd.DataFrame:
    """Predict house prices using the pre-trained model."""
    cat_features = [
        "MSZoning", "Street", "LotConfig",
        "Neighborhood", "Condition1",
    ]
    num_features = [
        "OverallQual", "GrLivArea",
        "TotRmsAbvGrd", "GarageCars",
    ]

    # Preprocess input data
    X_cat, X_num = preprocess_features(
        input_data, cat_features, num_features,
        cat_imputer, encoder, scaler
    )
    X_transformed = np.hstack([X_cat, X_num])

    # Feature mismatch check
    expected, actual = 16, X_transformed.shape[1]
    if actual != expected:
        print(f"Warning: Expected {expected} features, got {actual}.")

    # Predict SalePrice
    y_pred = model.predict(X_transformed)
    input_data["SalePrice"] = np.maximum(0, y_pred)

    return input_data[["Id", "SalePrice"]]


if __name__ == "__main__":
    input_data = pd.read_csv(
        "/Users/purnimaprabha/dsp-purnima-prabha/house_prices/test.csv"
    )
    print(make_predictions(input_data))
>>>>>>> pw2
