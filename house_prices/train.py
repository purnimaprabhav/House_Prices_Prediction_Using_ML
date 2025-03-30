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
