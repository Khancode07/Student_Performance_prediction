"""
Module 1: Student GPA Prediction
Predicts a student's GPA using demographic, parental, and behavioral features
using a Random Forest Regressor.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
DATA_PATH = "Student_performance_data _.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Feature Selection
# ---------------------------------------------------------
X = df.drop(columns=["StudentID", "GPA", "GradeClass"])
y = df["GPA"]

print("\nFeatures used:", list(X.columns))

# ---------------------------------------------------------
# 3. Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Train Model
# ---------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# ---------------------------------------------------------
# 6. Feature Importance
# ---------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print("\n--- Feature Importance ---")
print(importance)

# ---------------------------------------------------------
# 7. Save Model
# ---------------------------------------------------------
joblib.dump(model, "gpa_prediction_model.pkl")
print("\nModel saved as gpa_prediction_model.pkl")