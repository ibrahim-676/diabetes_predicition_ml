import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('diabetes.csv')

# Handle missing data
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[columns_with_zeros] = imputer.fit_transform(df[columns_with_zeros])

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Comprehensive Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                  Predicted")
print(f"                  Non-Diabetic  Diabetic")
print(f"Actual Non-Diabetic      {cm[0][0]}         {cm[0][1]}")
print(f"Actual Diabetic          {cm[1][0]}         {cm[1][1]}")

# Clinical interpretation
tn, fp, fn, tp = cm.ravel()
print(f"\nClinical Breakdown:")
print(f"Correctly identified non-diabetic: {tn}")
print(f"Incorrectly predicted diabetic: {fp}")
print(f"Missed diabetic cases: {fn}")
print(f"Correctly identified diabetic: {tp}")
print(f"\nDiabetic Detection Rate: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")