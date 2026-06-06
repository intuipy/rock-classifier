# Rock classification model v1.0
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv('data/rock_classification.csv')

# Remove missing resistivity values
data = data[data['RES'] != -999.25].copy()

# Drop any remaining NaN values
data = data.dropna(subset=['GR', 'RES', 'PHIE', 'RHOB', 'Unique_Class_Num'])

# Reset index
data = data.reset_index(drop=True)

# Use only the 4 input features
features = ['GR', 'RES', 'PHIE', 'RHOB']
X = data[features]
y = data['Unique_Class_Num']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(rf_model, 'model/rf_rock_classifier.pkl')
print('Model saved to model/rf_rock_classifier.pkl')