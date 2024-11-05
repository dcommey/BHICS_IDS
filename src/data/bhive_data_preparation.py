import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return pd.concat([train_data, test_data], axis=0, ignore_index=True)

def preprocess_data(data):
    # Convert attack types to binary classification
    data['Label'] = data['Label'].apply(lambda x: 0 if x == 0 else 1)
    
    return data

def extract_features(data):
    feature_columns = [
        'Max Packet Length', 'Fwd Packet Length Max', 'Packet Length Mean',
        'Average Packet Size', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Min Packet Length', 'Avg Fwd Segment Size', 'Flow Bytes/s',
        'Total Length of Fwd Packets', 'Subflow Fwd Bytes', 'Flow Duration',
        'Fwd IAT Total', 'Fwd Packets/s', 'Flow IAT Max', 'Flow IAT Mean',
        'Flow IAT Std', 'Fwd IAT Std', 'Fwd IAT Mean', 'Fwd Header Length',
        'Flow Packets/s', 'Init_Win_bytes_forward', 'Total Fwd Packets'
    ]
    
    existing_columns = [col for col in feature_columns if col in data.columns]
    missing_columns = set(feature_columns) - set(existing_columns)
    if missing_columns:
        print(f"Warning: The following columns are missing from the dataset: {missing_columns}")
    
    X = data[existing_columns]
    y = data['Label']
    
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def balance_classes(X, y):
    # Define the resampling pipeline
    over = SMOTE(sampling_strategy=0.5)  # Increase minority class to 50% of majority class
    under = RandomUnderSampler(sampling_strategy=0.75)  # Reduce majority class to have 25% more samples than minority class
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    
    # Fit and apply the pipeline
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    
    return X_resampled, y_resampled

def prepare_data_for_bhive(train_path, test_path, output_path):
    # Load and preprocess data
    data = load_data(train_path, test_path)
    data = preprocess_data(data)
    
    # Extract features and target
    X, y = extract_features(data)
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Balance classes
    X_balanced, y_balanced = balance_classes(X_scaled, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # Save prepared data
    np.savez(output_path, 
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             feature_names=X.columns.values)
    
    print(f"Prepared data saved to {output_path}")
    return scaler

# Usage
train_path = 'data/ddos_data_train.csv'
test_path = 'data/ddos_data_test.csv'
output_path = 'data/prepared_data_bhive.npz'

scaler = prepare_data_for_bhive(train_path, test_path, output_path)

# Save the scaler for later use
import joblib
joblib.dump(scaler, 'data/feature_scaler.joblib')

print("Data preparation completed successfully.")