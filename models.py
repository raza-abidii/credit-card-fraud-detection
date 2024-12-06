import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load

def train_and_save_model(dataset_path, model_path, scaler_path):
    data = pd.read_csv(dataset_path)
    print("Available columns:", data.columns.tolist())
    numeric_features = [
        'amt',
        'lat',
        'long',
        'city_pop',
        'merch_lat',
        'merch_long'
    ]
    categorical_features = ['category']
    X = pd.DataFrame()
    
    for col in numeric_features:
        if col in data.columns:
            X[col] = data[col]
        else:
            raise ValueError(f"Required column '{col}' not found in dataset")
        
    for col in categorical_features:
        if col in data.columns:
            X[col] = data[col]
    
    print("\nUsing features:", X.columns.tolist())

    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, prefix_sep='_')
    
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    if 'is_fraud' in data.columns:
        y = data['is_fraud']
    else:
        raise ValueError("Column 'is_fraud' not found in dataset")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    dump(model, model_path)
    dump(scaler, scaler_path)
    
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
    model = load(model_path)
    scaler = load(scaler_path)
    feature_config = load(feature_config_path)
    numeric_features = feature_config['numeric_features']
    categorical_features = feature_config['categorical_features']
    required_features = feature_config['feature_names']

    X = pd.DataFrame()

    for col in numeric_features:
        if col not in data:
            raise ValueError(f"Required numeric feature '{col}' not found in input data")
        X[col] = data[col]
    
    for col in categorical_features:
        if col in data:
            X[col] = data[col]
    
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, prefix_sep='_')
    
    for feature in required_features:
        if feature not in X.columns:
            X[feature] = 0  
            
    X = X[required_features]
    
    X[numeric_features] = scaler.transform(X[numeric_features])
    
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    train_and_save_model('./data/fraudTrain.csv', 'saved_model.pkl', 'scaler.pkl')