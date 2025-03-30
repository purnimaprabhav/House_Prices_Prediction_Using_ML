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
