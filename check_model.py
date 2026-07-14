import joblib

model = joblib.load("module1_gpa_prediction/gpa_prediction_model.pkl")
print("Model type:", type(model))
print(model)

if hasattr(model, "feature_names_in_"):
    print("\nFeatures used:", list(model.feature_names_in_))