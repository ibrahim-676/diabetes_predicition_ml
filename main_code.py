import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('diabetes.csv')

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
class_dist = df['Outcome'].value_counts()
print(f"Non-diabetic: {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
print(f"Diabetic: {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")

# Identify columns with missing values (zeros)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\n--- Missing Data Analysis (Zeros) ---")
for col in columns_with_zeros:
    zero_count = (df[col] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    print(f"{col}: {zero_count} zeros ({zero_pct:.1f}%)")

# Replace zeros with NaN and impute
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[columns_with_zeros] = imputer.fit_transform(df[columns_with_zeros])

print("\n--- After Imputation ---")
for col in columns_with_zeros:
    print(f"{col}: median = {df[col].median():.2f}")

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")