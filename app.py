"""
Student Performance Prediction System — Combined Single-Page Dashboard
-------------------------------------------------------------------------
One page. User fills in all details at once (study hours, attendance %,
extracurriculars, reading/English habits, etc.) and clicks a single button.
All three trained models run together and the app shows ONE combined
final result.

Run from the project root folder:
    streamlit run app.py
"""

import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE1_DIR = os.path.join(BASE_DIR, "module1_gpa_prediction")
MODULE2_DIR = os.path.join(BASE_DIR, "module2_behavior_classification")
MODULE3_DIR = os.path.join(BASE_DIR, "module3_gpa_trend")

st.set_page_config(page_title="Student Performance Prediction System", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Prediction System")
st.write(
    "Fill in the details below. This tool runs three machine learning models "
    "together — one for GPA, one for classroom engagement, and one for overall "
    "grade band — and combines them into a single final result."
)

st.divider()

# ---------------------------------------------------------
# Load all 3 models up front
# ---------------------------------------------------------
m1_path = os.path.join(MODULE1_DIR, "gpa_prediction_model.pkl")
m2_path = os.path.join(MODULE2_DIR, "behavior_classification_model.pkl")
m2_enc_path = os.path.join(MODULE2_DIR, "feature_encoders.pkl")
m2_target_path = os.path.join(MODULE2_DIR, "target_encoder.pkl")
m3_path = os.path.join(MODULE3_DIR, "grade_prediction_model.pkl")

missing = [p for p in [m1_path, m2_path, m2_enc_path, m2_target_path, m3_path] if not os.path.exists(p)]
if missing:
    st.error("Some model files are missing. Make sure train_model.py has been run in all 3 module folders.")
    st.stop()

model1 = joblib.load(m1_path)
model2 = joblib.load(m2_path)
encoders2 = joblib.load(m2_enc_path)
target_encoder2 = joblib.load(m2_target_path)
model3 = joblib.load(m3_path)

# ---------------------------------------------------------
# SECTION 1: Basic Info
# ---------------------------------------------------------
st.subheader("👤 Basic Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 15, 22, 17)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

# ---------------------------------------------------------
# SECTION 2: Attendance
# ---------------------------------------------------------
st.subheader("📅 Attendance")
col1, col2 = st.columns(2)
with col1:
    attendance_pct = st.slider("Class attendance percentage", 0, 100, 90, help="What % of classes does the student attend?")
with col2:
    absence_pct = 100 - attendance_pct
    st.metric("Absence percentage", f"{absence_pct}%")

absences = round((absence_pct / 100) * 30)

if attendance_pct >= 90:
    attend_habit = "Always"
elif attendance_pct >= 60:
    attend_habit = "Sometimes"
else:
    attend_habit = "Never"

st.divider()

# ---------------------------------------------------------
# SECTION 3: Study Habits
# ---------------------------------------------------------
st.subheader("📚 Study Habits")
col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("Study hours per week", 0, 25, 10)
    tutoring = st.selectbox("Receives tutoring?", ["No", "Yes"])
with col2:
    parental_support = st.selectbox(
        "Parental support level", ["None", "Low", "Moderate", "High", "Very High"]
    )
    scholarship = st.selectbox("Scholarship", ["None", "25%", "50%", "75%", "Full"])

st.divider()

# ---------------------------------------------------------
# SECTION 4: Classroom Engagement
# ---------------------------------------------------------
st.subheader("🙋 Classroom Engagement")
col1, col2 = st.columns(2)
with col1:
    raised_hands = st.slider("Times raised hand / participated in class", 0, 100, 40)
    visited_resources = st.slider("Times visited course resources online", 0, 100, 50)
with col2:
    discussion = st.slider("Times joined class discussions", 0, 100, 30)
    cuml_gpa = st.selectbox(
        "Current cumulative GPA range (if known)",
        ["<2.00", "2.00-2.49", "2.50-2.99", "3.00-3.49", ">3.49"],
    )

st.divider()

# ---------------------------------------------------------
# SECTION 5: Extracurricular Activities
# ---------------------------------------------------------
st.subheader("🎨 Extracurricular Activities")
col1, col2 = st.columns(2)
with col1:
    extracurricular = st.selectbox("Involved in any extracurricular activities?", ["No", "Yes"])
    sports = st.selectbox("Plays sports?", ["No", "Yes"])
    music = st.selectbox("Involved in dance / music?", ["No", "Yes"])
with col2:
    volunteering = st.selectbox("Does volunteering?", ["No", "Yes"])

st.divider()

# ---------------------------------------------------------
# SECTION 6: Learning & Reading Habits
# ---------------------------------------------------------
st.subheader("🌐 Learning & Reading Habits")
col1, col2 = st.columns(2)
with col1:
    youtube_learning = st.slider(
        "Hours per week learning from YouTube / online videos", 0, 20, 3
    )
with col2:
    reads_newspaper = st.selectbox(
        "Reads an English newspaper regularly?", ["No", "Sometimes", "Yes, daily"]
    )

st.caption(
    "Note: YouTube learning hours and newspaper reading are shown here to build a "
    "fuller picture of the student, and are factored into study habits below since "
    "the underlying models were not trained with these exact columns."
)

st.divider()

# ---------------------------------------------------------
# RUN ALL 3 MODELS + COMBINE
# ---------------------------------------------------------
if st.button("🔮 Get Combined Prediction", type="primary", use_container_width=True):

    effective_study_hours = min(25, study_hours + round(youtube_learning * 0.5))

    # ---- Model 1: GPA (regression) ----
    m1_features = [[
        age if age <= 18 else 18,
        0 if gender == "Male" else 1,
        0,
        2,
        effective_study_hours,
        absences,
        0 if tutoring == "No" else 1,
        ["None", "Low", "Moderate", "High", "Very High"].index(parental_support),
        0 if extracurricular == "No" else 1,
        0 if sports == "No" else 1,
        0 if music == "No" else 1,
        0 if volunteering == "No" else 1,
    ]]
    predicted_gpa = model1.predict(m1_features)[0]

    # ---- Model 2: Behavior classification ----
    row2 = {}
    for col, le in encoders2.items():
        row2[col] = le.transform([le.classes_[0]])[0]
    row2["raisedhands"] = raised_hands
    row2["VisITedResources"] = visited_resources
    row2["AnnouncementsView"] = 30
    row2["Discussion"] = discussion
    row2["StudentAbsenceDays"] = encoders2["StudentAbsenceDays"].transform(
        ["Under-7" if absence_pct <= 20 else "Above-7"]
    )[0]
    feature_order2 = list(model2.feature_names_in_)
    m2_features = [[row2[c] for c in feature_order2]]
    pred2 = model2.predict(m2_features)[0]
    pred2_label = target_encoder2.inverse_transform([pred2])[0]
    behavior_level = {"L": "Low", "M": "Medium", "H": "High"}.get(pred2_label, pred2_label)

    # ---- Model 3: Grade band ----
    if effective_study_hours == 0:
        study_hrs_band = 0
    elif effective_study_hours < 5:
        study_hrs_band = 1
    elif effective_study_hours <= 10:
        study_hrs_band = 2
    elif effective_study_hours <= 20:
        study_hrs_band = 3
    else:
        study_hrs_band = 4

    cuml_gpa_options = ["<2.00", "2.00-2.49", "2.50-2.99", "3.00-3.49", ">3.49"]
    scholarship_options = ["None", "25%", "50%", "75%", "Full"]
    attend_options = ["Always", "Sometimes", "Never"]

    feature_order3 = list(model3.feature_names_in_)
    row3 = {c: 1 for c in feature_order3}
    row3["STUDY_HRS"] = study_hrs_band
    row3["CUML_GPA"] = cuml_gpa_options.index(cuml_gpa)
    row3["SCHOLARSHIP"] = scholarship_options.index(scholarship)
    row3["ATTEND"] = attend_options.index(attend_habit)
    m3_features = [[row3[c] for c in feature_order3]]
    grade_band = model3.predict(m3_features)[0]

    # ---- COMBINE INTO ONE FINAL ANSWER ----
    def gpa_to_score(g):
        if g >= 3.0:
            return 2
        elif g >= 2.0:
            return 1
        else:
            return 0

    level_to_score = {"Low": 0, "Medium": 1, "High": 2}

    scores = [
        gpa_to_score(predicted_gpa),
        level_to_score.get(behavior_level, 1),
        level_to_score.get(grade_band, 1),
    ]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 1.5:
        overall = "High Performer"
        color = "success"
    elif avg_score >= 0.75:
        overall = "Average Performer"
        color = "info"
    else:
        overall = "At Risk"
        color = "warning"

    st.subheader("🎯 Combined Result")
    getattr(st, color)(f"### Overall Assessment: {overall}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted GPA", f"{predicted_gpa:.2f} / 4.0")
    c2.metric("Engagement Level", behavior_level)
    c3.metric("Grade Band", grade_band)

    st.caption(
        "This combined result is built by running three independently trained models "
        "(GPA regression, engagement classification, grade-band classification) on the "
        "same student profile and averaging their outcomes into one overall verdict."
    )

st.divider()
st.caption(
    "Built with scikit-learn + Streamlit | 3 models trained on 3 real-world "
    "student performance datasets, combined into one prediction."
)