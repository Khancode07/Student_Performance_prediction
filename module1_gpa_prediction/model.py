"""
Module 1: Student GPA Prediction
Predicts a student's final grade (G3, scale 0-20) using demographic,
family, lifestyle, and academic history features, combining the Math
and Portuguese course datasets from the UCI Student Performance dataset,
using a Random Forest Regressor.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. Load Data (combine Math + Portuguese datasets)
# ---------------------------------------------------------
df_math = pd.read_excel("Maths.csv")
df_por = pd.read_excel("Portuguese.csv")

df = pd.concat([df_math, df_por], ignore_index=True)
print(f"Maths: {df_math.shape[0]} rows | Portuguese: {df_por.shape[0]} rows | Combined: {df.shape[0]} rows")
print(df.head())

# ---------------------------------------------------------
# 2. Encode Categorical Columns
# ---------------------------------------------------------
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print("\nCategorical columns to encode:", categorical_cols)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------
# 3. Feature Selection
# ---------------------------------------------------------
# G1 and G2 (earlier grading periods) are dropped so the model predicts
# final grade from lifestyle/demographic factors alone, not just from
# earlier grades -- a more meaningful prediction task.
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

print("\nFeatures used:", list(X.columns))

# ---------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 5. Train Model
# ---------------------------------------------------------
model = RandomForestRegressor(
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

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
accuracy_percent = r2 * 100

print("\n--- Model Performance ---")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print(f"Model Accuracy (R² based): {accuracy_percent:.2f}%")

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
joblib.dump(model, "gpa_prediction_model.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
print("\nModel and encoders saved.")