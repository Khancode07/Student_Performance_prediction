"""
Unified Student Performance Prediction System
------------------------------------------------
This is the main entry point of the project. It lets the user choose
what type of student they want to predict for (School / College / University),
then loads the correct trained model for that type and asks the relevant
questions to produce a prediction.

Run this from the project root folder:
    python predict.py

Requirements: each module's train_model.py must have been run first so that
the .pkl model files exist inside their respective folders.
"""

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODULE1_DIR = os.path.join(BASE_DIR, "module1_gpa_prediction")
MODULE2_DIR = os.path.join(BASE_DIR, "module2_behavior_classification")
MODULE3_DIR = os.path.join(BASE_DIR, "module3_gpa_trend")


def ask_number(prompt, min_val=None, max_val=None):
    """Ask the user for a number, keep asking until valid."""
    while True:
        try:
            val = float(input(prompt))
            if min_val is not None and val < min_val:
                print(f"  Please enter a value >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Please enter a value <= {max_val}")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")


def ask_choice(prompt, options):
    """Ask the user to choose from a list of options, return the index."""
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  {i} - {opt}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 0 <= choice < len(options):
                return choice
        except ValueError:
            pass
        print("  Invalid choice, try again.")


# ---------------------------------------------------------
# MODULE 1: School Student GPA Prediction
# ---------------------------------------------------------
def run_module1():
    model_path = os.path.join(MODULE1_DIR, "gpa_prediction_model.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py inside module1_gpa_prediction first.")
        return

    model = joblib.load(model_path)

    print("\n--- School Student GPA Prediction ---")
    age = ask_number("Age (15-18): ", 15, 18)
    gender = ask_choice("Gender:", ["Male", "Female"])
    ethnicity = ask_choice("Ethnicity group:", ["Group 0", "Group 1", "Group 2", "Group 3", "Group 4"])
    parental_edu = ask_choice("Parental Education:", ["None", "High School", "Some College", "Bachelor's", "Higher"])
    study_time = ask_number("Study time per week (hours, 0-20): ", 0, 20)
    absences = ask_number("Number of absences (0-30): ", 0, 30)
    tutoring = ask_choice("Receives tutoring?", ["No", "Yes"])
    parental_support = ask_choice("Parental Support level:", ["None", "Low", "Moderate", "High", "Very High"])
    extracurricular = ask_choice("Extracurricular activities?", ["No", "Yes"])
    sports = ask_choice("Plays sports?", ["No", "Yes"])
    music = ask_choice("Involved in music?", ["No", "Yes"])
    volunteering = ask_choice("Does volunteering?", ["No", "Yes"])

    features = [[age, gender, ethnicity, parental_edu, study_time, absences,
                 tutoring, parental_support, extracurricular, sports, music, volunteering]]

    predicted_gpa = model.predict(features)[0]
    print(f"\nPredicted GPA: {predicted_gpa:.2f} (scale 0.0 - 4.0)")


# ---------------------------------------------------------
# MODULE 2: School Student Behavior Classification
# ---------------------------------------------------------
def run_module2():
    model_path = os.path.join(MODULE2_DIR, "behavior_classification_model.pkl")
    encoders_path = os.path.join(MODULE2_DIR, "feature_encoders.pkl")
    target_encoder_path = os.path.join(MODULE2_DIR, "target_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, encoders_path, target_encoder_path]):
        print("Model not found. Run train_model.py inside module2_behavior_classification first.")
        return

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    target_encoder = joblib.load(target_encoder_path)

    print("\n--- School Student Behavior-Based Performance Prediction ---")
    raised_hands = ask_number("Times raised hand in class (0-100): ", 0, 100)
    visited_resources = ask_number("Times visited course resources (0-100): ", 0, 100)
    announcements = ask_number("Times viewed announcements (0-100): ", 0, 100)
    discussion = ask_number("Times participated in discussion (0-100): ", 0, 100)
    absence_days = ask_choice("Absence days:", ["Under-7", "Above-7"])

    row = {}
    for col, le in encoders.items():
        row[col] = le.transform([le.classes_[0]])[0]

    row["raisedhands"] = raised_hands
    row["VisITedResources"] = visited_resources
    row["AnnouncementsView"] = announcements
    row["Discussion"] = discussion
    row["StudentAbsenceDays"] = encoders["StudentAbsenceDays"].transform(
        ["Under-7" if absence_days == 0 else "Above-7"]
    )[0]

    feature_order = list(model.feature_names_in_)
    features = [[row[col] for col in feature_order]]

    pred = model.predict(features)[0]
    pred_label = target_encoder.inverse_transform([pred])[0]
    label_map = {"L": "Low", "M": "Medium", "H": "High"}
    print(f"\nPredicted Performance Level: {label_map.get(pred_label, pred_label)}")
    print("(Note: other fields used typical/default values since this quick "
          "predictor only asks about class engagement.)")


# ---------------------------------------------------------
# MODULE 3: University Student Grade Prediction
# ---------------------------------------------------------
def run_module3():
    model_path = os.path.join(MODULE3_DIR, "grade_prediction_model.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py inside module3_gpa_trend first.")
        return

    model = joblib.load(model_path)

    print("\n--- University Student Grade Band Prediction ---")
    study_hrs = ask_choice("Weekly study hours:", ["None", "<5 hrs", "6-10 hrs", "11-20 hrs", ">20 hrs"])
    cuml_gpa = ask_choice("Cumulative GPA range:", ["<2.00", "2.00-2.49", "2.50-2.99", "3.00-3.49", ">3.49"])
    scholarship = ask_choice("Scholarship type:", ["None", "25%", "50%", "75%", "Full"])
    attend = ask_choice("Class attendance:", ["Always", "Sometimes", "Never"])

    feature_order = list(model.feature_names_in_)
    row = {col: 1 for col in feature_order}
    row["STUDY_HRS"] = study_hrs
    row["CUML_GPA"] = cuml_gpa
    row["SCHOLARSHIP"] = scholarship
    row["ATTEND"] = attend

    features = [[row[col] for col in feature_order]]
    pred_band = model.predict(features)[0]
    print(f"\nPredicted Grade Band: {pred_band}")
    print("(Note: other fields used neutral default values since this quick "
          "predictor only asks about the most important factors.)")


# ---------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------
def main():
    print("=" * 55)
    print("   STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 55)
    print("\nWhat type of student are you predicting for?")
    print("  1 - School Student  -> GPA Prediction")
    print("  2 - School Student  -> Behavior/Engagement-based Classification")
    print("  3 - University Student -> Grade Band Prediction")
    print("  0 - Exit")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        run_module1()
    elif choice == "2":
        run_module2()
    elif choice == "3":
        run_module3()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()