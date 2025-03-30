import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from house_prices.preprocess import clean_data, preprocess_features, encode_features, scale_features

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message=(
        "The number of features does not match the model's expected input!"
    ),
)

# Add custom module path
sys.path.append("/Users/purnimaprabha/dsp-purnima-prabha")

# Model and preprocessing objects
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

    model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'standard_scaler.pkl'))
    encoder = joblib.load(os.path.join(models_dir, 'one_hot_encoder.pkl'))
    cat_imputer = joblib.load(os.path.join(models_dir, 'cat_imputer.pkl'))
    
    return model, scaler, encoder, cat_imputer

def make_predictions(input_data: pd.DataFrame) -> pd.DataFrame:
    """Make predictions on new data.

    Args:
        input_data: DataFrame containing features to predict on

    Returns:
        DataFrame with predictions
    """
    # Load model and preprocessing objects
    model, scaler, encoder, cat_imputer = load_objects()

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

    # Feature mismatch check (optional)
    expected, actual = 16, processed_data.shape[1]  # Adjust the expected value if needed
    if actual != expected:
        print(f"Warning: Expected {expected} features, got {actual}.")

    # Make predictions
    y_pred = model.predict(processed_data)
    input_data["SalePrice"] = np.maximum(0, y_pred)

    return input_data[["Id", "SalePrice"]]


if __name__ == "__main__":
    input_data = pd.read_csv("/Users/purnimaprabha/dsp-purnima-prabha/house_prices/test.csv")
    print(make_predictions(input_data))
