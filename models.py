import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load

def train_and_save_model(dataset_path, model_path, scaler_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Print column names and first few rows to verify the structure
    print("Available columns:", data.columns.tolist())
    
    # Define features based on the actual column names in fraudTrain.csv
    numeric_features = [
        'amt',
        'lat',
        'long',
        'city_pop',
        'merch_lat',
        'merch_long'
    ]
    categorical_features = ['category']
    
    # Create feature matrix X
    X = pd.DataFrame()
    
    # Add numeric features
    for col in numeric_features:
        if col in data.columns:
            X[col] = data[col]
        else:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Add categorical features
    for col in categorical_features:
        if col in data.columns:
            X[col] = data[col]
    
    print("\nUsing features:", X.columns.tolist())
    
    # Convert categorical variables using one-hot encoding
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, prefix_sep='_')
    
    # Get list of all feature names after one-hot encoding
    feature_names = X.columns.tolist()
    
    # Standardize numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Get the target variable
    if 'is_fraud' in data.columns:
        y = data['is_fraud']
    else:
        raise ValueError("Column 'is_fraud' not found in dataset")

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model and scaler
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    # Save the feature configuration
    feature_config = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_names': feature_names
    }
    dump(feature_config, 'feature_config.pkl')
    
    print("\nModel training completed and saved successfully!")
    print("Required features for prediction:")
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("All features after encoding:", feature_names)

def predict_fraud(data, model_path='saved_model.pkl', scaler_path='scaler.pkl', feature_config_path='feature_config.pkl'):
    """
    Make predictions using the trained model
    """
    # Load the model, scaler and feature configuration
    model = load(model_path)
    scaler = load(scaler_path)
    feature_config = load(feature_config_path)
    
    # Extract features from configuration
    numeric_features = feature_config['numeric_features']
    categorical_features = feature_config['categorical_features']
    required_features = feature_config['feature_names']
    
    # Prepare the input data
    X = pd.DataFrame()
    
    # Add numeric features
    for col in numeric_features:
        if col not in data:
            raise ValueError(f"Required numeric feature '{col}' not found in input data")
        X[col] = data[col]
    
    # Add categorical features
    for col in categorical_features:
        if col in data:
            X[col] = data[col]
    
    # Apply one-hot encoding to categorical features
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, prefix_sep='_')
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in X.columns:
            X[feature] = 0  # Add missing dummy variables with 0
    
    # Reorder columns to match training data
    X = X[required_features]
    
    # Scale numeric features
    X[numeric_features] = scaler.transform(X[numeric_features])
    
    # Make prediction
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    train_and_save_model('./data/fraudTrain.csv', 'saved_model.pkl', 'scaler.pkl')