"""
Module 3: University Student Grade Prediction
Predicts a university student's final course grade band (Low/High)
using lifestyle, study habits, and socio-economic features,
using a Random Forest Classifier.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
DATA_PATH = "student_prediction.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Group Grades into 2 Broader Categories
# ---------------------------------------------------------
# The raw GRADE column has 8 fine-grained classes (0-7), and with only 145
# rows total, a 3-way split (Low/Medium/High) left the Medium class with too
# few samples (31) to learn reliably -- confirmed via testing with both
# class_weight="balanced" and SMOTE oversampling, neither of which resolved
# it. Simplified to a 2-way split (Low/High) which better matches the data's
# natural separability.
def grade_band(g):
    if g <= 3:
        return "Low"
    else:
        return "High"

df["GRADE_BAND"] = df["GRADE"].apply(grade_band)
print("\nGrade band distribution:")
print(df["GRADE_BAND"].value_counts())

# ---------------------------------------------------------
# 3. Features / Target Split
# ---------------------------------------------------------
X = df.drop(columns=["STUDENTID", "GRADE", "GRADE_BAND"])
y = df["GRADE_BAND"]

print("\nFeatures used:", list(X.columns))

# ---------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4.5 Handle Class Imbalance with SMOTE
# ---------------------------------------------------------
smote = SMOTE(random_state=42, k_neighbors=3)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE, training class distribution:")
print(y_train.value_counts())

# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 7. Feature Importance
# ---------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print("\n--- Top 10 Feature Importance ---")
print(importance.head(10))

# ---------------------------------------------------------
# 8. Save Model
# ---------------------------------------------------------
joblib.dump(model, "grade_prediction_model.pkl")
print("\nModel saved as grade_prediction_model.pkl")