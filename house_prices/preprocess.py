"""
This module contains functions for data preprocessing and feature engineering.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def initialize_transformers() -> Dict:
    """
    Initialize all the preprocessing transformers.

    Returns:
        dict: Dictionary containing all initialized transformers
    """
    return {
        'num_imputer': SimpleImputer(strategy='median'),
        'cat_imputer': SimpleImputer(strategy='constant', fill_value='missing'),
        'scaler': StandardScaler(),
        'encoder': OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    }


def process_numerical_features(
    data: pd.DataFrame,
    numerical_features: List[str],
    num_imputer: SimpleImputer,
    scaler: StandardScaler,
    fit: bool = False
) -> pd.DataFrame:
    """
    Process numerical features with imputation and scaling.

    Args:
        data: Input DataFrame containing numerical features
        numerical_features: List of numerical feature names
        num_imputer: Numerical imputer object
        scaler: StandardScaler object
        fit: Whether to fit the transformers or just transform

    Returns:
        pd.DataFrame: Processed numerical features
    """
    X_num = data[numerical_features].copy()
    
    if fit:
        X_num_imputed = num_imputer.fit_transform(X_num)
        X_num_scaled = scaler.fit_transform(X_num_imputed)
    else:
        X_num_imputed = num_imputer.transform(X_num)
        X_num_scaled = scaler.transform(X_num_imputed)
    
    return pd.DataFrame(X_num_scaled, columns=numerical_features, index=data.index)


def process_categorical_features(
    data: pd.DataFrame,
    categorical_features: List[str],
    cat_imputer: SimpleImputer,
    encoder: OneHotEncoder,
    fit: bool = False
) -> pd.DataFrame:
    """
    Process categorical features with imputation and encoding.

    Args:
        data: Input DataFrame containing categorical features
        categorical_features: List of categorical feature names
        cat_imputer: Categorical imputer object
        encoder: OneHotEncoder object
        fit: Whether to fit the transformers or just transform

    Returns:
        pd.DataFrame: Processed categorical features
    """
    X_cat = data[categorical_features].copy()
    
    if fit:
        X_cat_imputed = cat_imputer.fit_transform(X_cat)
        X_cat_encoded = encoder.fit_transform(X_cat_imputed)
    else:
        X_cat_imputed = cat_imputer.transform(X_cat)
        X_cat_encoded = encoder.transform(X_cat_imputed)
    
    feature_names = encoder.get_feature_names_out(categorical_features)
    return pd.DataFrame(X_cat_encoded, columns=feature_names, index=data.index)


def preprocess_data(
    data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    transformers: Dict,
    fit: bool = False
) -> pd.DataFrame:
    """
    Preprocess data by applying all transformations.

    Args:
        data: Input DataFrame
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        transformers: Dictionary containing all transformers
        fit: Whether to fit the transformers or just transform

    Returns:
        pd.DataFrame: Processed features
    """
    # Process numerical features
    X_num_processed = process_numerical_features(
        data,
        numerical_features,
        transformers['num_imputer'],
        transformers['scaler'],
        fit
    )
    
    # Process categorical features
    X_cat_processed = process_categorical_features(
        data,
        categorical_features,
        transformers['cat_imputer'],
        transformers['encoder'],
        fit
    )
    
    # Combine features
    return pd.concat([X_num_processed, X_cat_processed], axis=1)
