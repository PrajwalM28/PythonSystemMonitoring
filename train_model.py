from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('synthetic_anomaly_data.csv')

# Select features
X = df[['cpu_usage', 'temperature']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train isolation forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)

# Save the model and scaler
joblib.dump(model, 'models/anomaly_detection.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model and scaler saved!")


