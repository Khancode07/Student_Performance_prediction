"""
Student Performance Prediction System — Minimal Dashboard
-------------------------------------------------------------------------
Clean, minimal UI: neutral palette, restrained typography, quiet layout.
Run from the project root folder:
    python -m streamlit run app.py
"""

import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE1_DIR = os.path.join(BASE_DIR, "module1_gpa_prediction")
MODULE2_DIR = os.path.join(BASE_DIR, "module2_behavior_classification")
MODULE3_DIR = os.path.join(BASE_DIR, "module3_gpa_trend")

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)
with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Minimal design system
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8A8F98;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #14161A;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 2.5rem;
        max-width: 560px;
    }

    /* Section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #14161A;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1.2rem 0;
    }

    /* Streamlit tabs, minimal underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B7280;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #14161A !important;
        font-weight: 600;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FAFAFA;
        border: 1px solid #ECECEC;
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #8A8F98; }

    /* Primary button */
    .stButton button {
        background: #14161A;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        letter-spacing: 0.01em;
    }
    .stButton button:hover { background: #2B2F36; color: white; }

    /* Result banner */
    .result-banner {
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        border-left: 3px solid;
    }
    .result-high  { background: #F4FAF6; color: #1F7A4D; border-color: #1F7A4D; }
    .result-avg   { background: #FEFAF2; color: #92650F; border-color: #C9860F; }
    .result-risk  { background: #FDF4F3; color: #A83A31; border-color: #C4453A; }

    .footer-note { color: #A0A4AB; font-size: 0.78rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">

<h1>🎓 Student Performance Prediction System</h1>

<p>
AI-Powered Student Performance Analytics Dashboard
</p>

<br>

<div style="display:flex;gap:18px;flex-wrap:wrap;">

<div style="background:rgba(255,255,255,.15);
padding:15px 22px;
border-radius:15px;
text-align:center;
min-width:140px;">

<h2 style="margin:0;">3</h2>
<p style="margin:0;">AI Models</p>

</div>

<div style="background:rgba(255,255,255,.15);
padding:15px 22px;
border-radius:15px;
text-align:center;
min-width:140px;">

<h2 style="margin:0;">20+</h2>
<p style="margin:0;">Input Features</p>

</div>

<div style="background:rgba(255,255,255,.15);
padding:15px 22px;
border-radius:15px;
text-align:center;
min-width:140px;">

<h2 style="margin:0;">Instant</h2>
<p style="margin:0;">Prediction</p>

</div>

<div style="background:rgba(255,255,255,.15);
padding:15px 22px;
border-radius:15px;
text-align:center;
min-width:140px;">

<h2 style="margin:0;">ML</h2>
<p style="margin:0;">Powered</p>

</div>

</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
m1_path = os.path.join(MODULE1_DIR, "gpa_prediction_model.pkl")
m2_path = os.path.join(MODULE2_DIR, "behavior_classification_model.pkl")
m2_enc_path = os.path.join(MODULE2_DIR, "feature_encoders.pkl")
m2_target_path = os.path.join(MODULE2_DIR, "target_encoder.pkl")
m3_path = os.path.join(MODULE3_DIR, "grade_prediction_model.pkl")

missing = [p for p in [m1_path, m2_path, m2_enc_path, m2_target_path, m3_path] if not os.path.exists(p)]
if missing:
    st.error("Some model files are missing. Run the training script in each module folder first.")
    with st.expander("Missing files"):
        for p in missing:
            st.code(p)
    st.stop()

model1 = joblib.load(m1_path)
model2 = joblib.load(m2_path)
encoders2 = joblib.load(m2_enc_path)
target_encoder2 = joblib.load(m2_target_path)
model3 = joblib.load(m3_path)

# ---------------------------------------------------------
# Input tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Basic", "Academics", "Engagement", "Activities"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 15, 22, 17)
        attendance_pct = st.slider("Attendance (%)", 0, 100, 90)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        absence_pct = 100 - attendance_pct
        st.metric("Absence rate", f"{absence_pct}%")

    absences = round((absence_pct / 100) * 30)
    if attendance_pct >= 90:
        attend_habit = "Always"
    elif attendance_pct >= 60:
        attend_habit = "Sometimes"
    else:
        attend_habit = "Never"

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study hours / week", 0, 25, 10)
        tutoring = st.selectbox("Receives tutoring?", ["No", "Yes"])
    with col2:
        parental_support = st.selectbox(
            "Parental support", ["None", "Low", "Moderate", "High", "Very High"]
        )
        scholarship = st.selectbox("Scholarship", ["None", "25%", "50%", "75%", "Full"])
        cuml_gpa = st.selectbox(
            "Cumulative GPA range", ["<2.00", "2.00-2.49", "2.50-2.99", "3.00-3.49", ">3.49"]
        )

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        raised_hands = st.slider("Hand raises / participation", 0, 100, 40)
        visited_resources = st.slider("Resource views", 0, 100, 50)
    with col2:
        discussion = st.slider("Discussion participation", 0, 100, 30)
        youtube_learning = st.slider("Self-study hours (video/online)", 0, 20, 3)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        extracurricular = st.selectbox("Extracurricular activities?", ["No", "Yes"])
        sports = st.selectbox("Plays sports?", ["No", "Yes"])
    with col2:
        music = st.selectbox("Music / dance?", ["No", "Yes"])
        volunteering = st.selectbox("Volunteering?", ["No", "Yes"])
        internet = st.selectbox("Internet at home?", ["Yes", "No"])

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
run_clicked = st.button("Get combined prediction", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running models..."):
        effective_study_hours = min(25, study_hours + round(youtube_learning * 0.5))

        # ---- Model 1: GPA (regression, original 12-feature model) ----
        m1_features = [[
            age if age <= 18 else 18,  # model trained on ages 15-18
            0 if gender == "Male" else 1,
            0,  # ethnicity - default
            2,  # parental education - default (Some College)
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
        predicted_cgpa = (predicted_gpa / 4.0) * 10.0  # convert 0-4 GPA scale to 0-10 CGPA scale

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

        def cgpa_to_score(g):
            if g >= 7.5:
                return 2
            elif g >= 5.0:
                return 1
            else:
                return 0

        level_to_score = {"Low": 0, "Medium": 1, "High": 2}
        scores = [
            cgpa_to_score(predicted_cgpa),
            level_to_score.get(behavior_level, 1),
            level_to_score.get(grade_band, 1),
        ]
        avg_score = sum(scores) / len(scores)

        if avg_score >= 1.5:
            overall = "High Performer"
            css_class = "result-high"
        elif avg_score >= 0.75:
            overall = "Average Performer"
            css_class = "result-avg"
        else:
            overall = "At Risk"
            css_class = "result-risk"

    st.markdown(f'<div class="result-banner {css_class}">{overall}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted GPA", f"{predicted_gpa:.2f} / 4.0", f"{predicted_cgpa:.2f} CGPA")
    c2.metric("Engagement", behavior_level)
    c3.metric("Grade band", grade_band)

    st.caption(
        "Combined from three independently trained models — final-grade regression, "
        "engagement classification, and grade-band classification."
    )

st.markdown(
    '<div class="footer-note">scikit-learn · Streamlit · 3 real-world datasets</div>',
    unsafe_allow_html=True,
)