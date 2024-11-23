import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filename):
    # Load the dataset
    df = pd.read_csv('synthetic_anomaly_data.csv')

    # Features and target
    X = df[['cpu_usage', 'temperature']]
    y = df['is_anomaly']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Preprocess and save
X_train, X_test, y_train, y_test, scaler = preprocess_data('data/system_health.csv')