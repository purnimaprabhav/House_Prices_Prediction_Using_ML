<<<<<<< HEAD
"""Module for model training functions."""
from typing import Dict, Tuple
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from .preprocess import clean_data, encode_features, scale_features


def split_data(
    df: pd.DataFrame,
    target_col: str = 'SalePrice',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        df: Input dataframe
        target_col: Target column
        test_size: Split ratio
        random_state: Seed value

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> RandomForestRegressor:
    """Train a Random Forest model.

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random state for reproducibility

    Returns:
        Trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with performance metrics
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {'rmse': rmse}


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """Build and train the model pipeline.

    Args:
        data: Input dataframe with features and target

    Returns:
        Dictionary with model performance metrics
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Split data first
    X_train, X_test, y_train, y_test = split_data(data)

    # Clean data
    X_train_cleaned = clean_data(X_train)
    X_test_cleaned = clean_data(X_test)

    # Get feature lists
    cat_cols = X_train_cleaned.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical features
    X_train_enc, encoder = encode_features(
        X_train_cleaned, cat_cols, fit=True
    )
    X_test_enc, _ = encode_features(
        X_test_cleaned, cat_cols, fit=False, encoder=encoder
    )

    # Scale numeric features
    X_train_proc, scaler = scale_features(
        X_train_enc, num_cols, fit=True
    )
    X_test_proc, _ = scale_features(
        X_test_enc, num_cols, fit=False, scaler=scaler
    )

    # Train model
    model = train_model(X_train_proc, y_train)

    # Save objects
    joblib.dump(encoder, os.path.join(models_dir, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(model, os.path.join(models_dir, 'model.joblib'))

    # Evaluate model
    performance = evaluate_model(model, X_test_proc, y_test)

    return performance
=======
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_features(
    X: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
    cat_imputer=None,
    encoder=None,
    scaler=None,
):
    """
    Impute, encode, and scale features.
    """
    cat_imputer = cat_imputer or SimpleImputer(strategy="most_frequent")
    encoder = encoder or OneHotEncoder(
        sparse_output=False, handle_unknown="ignore")
    scaler = scaler or StandardScaler()

    X_cat_imputed = cat_imputer.fit_transform(X[cat_features])
    X_cat_encoded = encoder.fit_transform(X_cat_imputed)
    X_num_scaled = scaler.fit_transform(X[num_features])

    return X_cat_encoded, X_num_scaled, cat_imputer, encoder, scaler


def build_model(
        training_data_df: pd.DataFrame,
        test_data_df: pd.DataFrame) -> dict:
    """
    Train a RandomForestRegressor and evaluate.
    """
    cat_features = [
        "MSZoning",
        "Street",
        "LotConfig",
        "Neighborhood",
        "Condition1"]
    num_features = ["OverallQual", "GrLivArea", "TotRmsAbvGrd", "GarageCars"]

    X_train, y_train = (
        training_data_df.drop(columns="SalePrice"),
        training_data_df["SalePrice"],
    )
    X_test, y_test = test_data_df.drop(
        columns="SalePrice"), test_data_df["SalePrice"]

    X_train_cat_encoded, X_train_num_scaled, cat_imputer, encoder, scaler = (
        preprocess_features(X_train, cat_features, num_features)
    )
    X_test_cat_encoded, X_test_num_scaled, _, _, _ = preprocess_features(
        X_test, cat_features, num_features, cat_imputer, encoder, scaler
    )

    # Match columns of test data to train data
    X_test_cat_encoded = pd.DataFrame(
        X_test_cat_encoded, columns=encoder.get_feature_names_out(cat_features)
    )
    X_test_cat_encoded = X_test_cat_encoded.reindex(
        columns=X_train_cat_encoded.columns, fill_value=0
    )

    # Combine features and train model
    X_train_transformed = np.hstack([X_train_cat_encoded, X_train_num_scaled])
    X_test_transformed = np.hstack([X_test_cat_encoded, X_test_num_scaled])

    model = RandomForestRegressor().fit(X_train_transformed, y_train)

    # Predict and evaluate
    y_test_pred = model.predict(X_test_transformed)
    mse, r2 = mean_squared_error(
        y_test, y_test_pred), r2_score(
        y_test, y_test_pred)

    # Save model and transformers
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(cat_imputer, "models/cat_imputer.pkl")
    joblib.dump(encoder, "models/one_hot_encoder.pkl")
    joblib.dump(scaler, "models/standard_scaler.pkl")

    return {"Mean Squared Error": mse, "R-squared": r2}
>>>>>>> pw2
