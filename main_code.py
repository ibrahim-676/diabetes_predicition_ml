import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-validation
print("\n5-Fold Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nTest Set Results:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# Generate visualizations
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Class Distribution
ax1 = fig.add_subplot(gs[0, 0])
class_dist = df['Outcome'].value_counts()
ax1.bar(['Non-Diabetic', 'Diabetic'], class_dist.values, color=['#4CAF50', '#F44336'])
ax1.set_title('Class Distribution', fontsize=10, fontweight='bold')
ax1.set_ylabel('Number of Patients', fontsize=9)
for i, v in enumerate(class_dist.values):
    ax1.text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=8)

# 2. Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'],
            annot_kws={'fontsize': 9})
ax2.set_title('Confusion Matrix', fontsize=10, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=9)
ax2.set_xlabel('Predicted', fontsize=9)
ax2.tick_params(labelsize=8)

# 3. ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'AUC = {roc_auc:.3f}')
ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
ax3.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
ax3.set_xlabel('False Positive Rate', fontsize=9)
ax3.set_ylabel('True Positive Rate', fontsize=9)
ax3.set_title('ROC Curve', fontsize=10, fontweight='bold')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(alpha=0.3)
ax3.tick_params(labelsize=8)

# 4. Feature Correlation Heatmap
ax4 = fig.add_subplot(gs[1, 0])
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax4,
            annot_kws={'fontsize': 7})
ax4.set_title('Feature Correlations', fontsize=10, fontweight='bold')
ax4.tick_params(labelsize=7)

# 5. Feature Importance
ax5 = fig.add_subplot(gs[1, 1:])
feature_names = X.columns
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

colors = ['#E63946' if x < 0 else '#06A77D' for x in feature_importance['Coefficient']]
bars = ax5.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors, edgecolor='black')
ax5.set_xlabel('Coefficient Value', fontsize=9)
ax5.set_title('Feature Importance (Higher = More Predictive)', fontsize=10, fontweight='bold')
ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax5.grid(axis='x', alpha=0.3)
ax5.tick_params(labelsize=8)

for i, (bar, coef) in enumerate(zip(bars, feature_importance['Coefficient'])):
    ax5.text(coef + (0.05 if coef > 0 else -0.05), i, f'{coef:.2f}', 
             va='center', ha='left' if coef > 0 else 'right', fontweight='bold', fontsize=7)

plt.savefig('figures_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")