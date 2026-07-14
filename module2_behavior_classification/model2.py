"""
Module 2: Student Behavior-Based Performance Classification
Classifies a student's performance level (Low / Medium / High) based on
classroom engagement behavior (raised hands, resource views, discussion, etc.)
using a Random Forest Classifier.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
DATA_PATH = "xAPI-Edu-Data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Encode Categorical Columns
# ---------------------------------------------------------
target_col = "Class"

categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove(target_col)

print("\nCategorical columns encoded:", categorical_cols)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target (L=Low, M=Medium, H=High)
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])
print("Class mapping:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

# ---------------------------------------------------------
# 3. Features / Target Split
# ---------------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
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
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 7. Feature Importance
# ---------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(importance)

# ---------------------------------------------------------
# 8. Save Model + Encoders
# ---------------------------------------------------------
joblib.dump(model, "behavior_classification_model.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("\nModel and encoders saved.")