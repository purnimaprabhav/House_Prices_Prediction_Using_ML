<<<<<<< HEAD
"""Module for data preprocessing and feature engineering functions."""
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the input dataframe by handling missing values.

    Args:
        df: Input dataframe to be cleaned

    Returns:
        Cleaned dataframe
    """
    df_cleaned = df.copy()

    # Fill numeric columns with median
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Fill categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

    return df_cleaned


def encode_features(
    df: pd.DataFrame,
    categorical_features: list[str],
    fit: bool = False,
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical features using OneHotEncoder.

    Args:
        df: Input dataframe
        categorical_features: List of categorical column names
        fit: Whether to fit the encoder or use pre-fitted
        encoder: Pre-fitted encoder (if fit is False)

    Returns:
        Tuple of encoded dataframe and fitted encoder
    """
    if not categorical_features:
        return df.copy(), encoder or OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(df[categorical_features])
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided when fit=False")
        # Handle unknown categories by replacing them with NaN
        df_copy = df.copy()
        for feature in categorical_features:
            known_categories = encoder.categories_[categorical_features.index(feature)]
            df_copy[feature] = df_copy[feature].apply(
                lambda x: x if x in known_categories else np.nan
            )
        encoded_features = encoder.transform(df_copy[categorical_features])

    feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=feature_names,
        index=df.index
    )

    # Drop original categorical columns and concat encoded ones
    result_df = pd.concat(
        [df.drop(columns=categorical_features), encoded_df],
        axis=1
    )

    return result_df, encoder


def scale_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    fit: bool = False,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler.

    Args:
        df: Input dataframe
        numeric_features: List of numeric column names
        fit: Whether to fit the scaler or use pre-fitted
        scaler: Pre-fitted scaler (if fit is False)

    Returns:
        Tuple of scaled dataframe and fitted scaler
    """
    if not numeric_features:
        return df.copy(), scaler or StandardScaler()

    if fit:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numeric_features])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        scaled_features = scaler.transform(df[numeric_features])

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=numeric_features,
        index=df.index
    )

    # Replace original numeric columns with scaled ones
    result_df = df.copy()
    result_df[numeric_features] = scaled_df

    return result_df, scaler
=======
import sys


sys.path.append("/Users/purnimaprabha/dsp-purnima-prabha")


def preprocess_features(
    input_data, cat_features, num_features, cat_imputer, encoder, scaler
):
    # Handle categorical features
    X_cat = input_data[cat_features]
    X_cat_imputed = cat_imputer.transform(X_cat)  # Use the imputer
    X_cat_encoded = encoder.transform(
        X_cat_imputed)  # Encode using one-hot encoder

    # Handle numerical features
    X_num = input_data[num_features]
    X_num_scaled = scaler.transform(X_num)  # Scale using the scaler

    return X_cat_encoded, X_num_scaled
>>>>>>> pw2
