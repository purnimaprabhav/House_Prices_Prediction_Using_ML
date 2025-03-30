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
