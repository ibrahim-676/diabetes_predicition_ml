import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('diabetes.csv')

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
class_dist = df['Outcome'].value_counts()
print(f"Non-diabetic: {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
print(f"Diabetic: {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")

# Handle missing data
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[columns_with_zeros] = imputer.fit_transform(df[columns_with_zeros])

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Feature Scaling ---")
print("Before scaling - Sample ranges:")
print(f"Glucose: {X_train['Glucose'].min():.2f} to {X_train['Glucose'].max():.2f}")
print(f"Pedigree: {X_train['DiabetesPedigreeFunction'].min():.2f} to {X_train['DiabetesPedigreeFunction'].max():.2f}")

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("After scaling - All features normalized to mean=0, std=1")

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")