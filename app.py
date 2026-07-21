"""
Student Performance Prediction System — Professional Single-Form Dashboard
-------------------------------------------------------------------------
Clean, professional UI: card-based sections, refined typography, subtle
shadows and dividers. Everything is on ONE page/form (no tabs) — fill in
top to bottom, then click one button to get the combined prediction.

Run from the project root folder:
    python -m streamlit run app.py
"""
from database import students
from database import predictions
from datetime import datetime
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
# Professional design system
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    :root {
        --ink: #14161A;
        --ink-soft: #4B5058;
        --muted: #8A8F98;
        --border: #E5E7EB;
        --surface: #FFFFFF;
        --surface-alt: #F7F8FA;
        --accent: #2F5D50;
        --accent-soft: #E9F1EE;
    }

    .main .block-container {
        max-width: 760px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */
    .header-wrap {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.8rem;
        margin-bottom: 2.2rem;
    }
    .kicker {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.6rem;
    }
    .doc-title {
        font-family: 'Source Serif 4', serif;
        font-size: 2.15rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.01em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }
    .doc-subtitle {
        color: var(--ink-soft);
        font-size: 0.97rem;
        line-height: 1.55;
        max-width: 600px;
        margin-bottom: 1.6rem;
    }
    .stat-row { display: flex; gap: 0; border-top: 1px solid var(--border); }
    .stat-cell {
        flex: 1;
        padding: 0.9rem 0 0 0;
        border-right: 1px solid var(--border);
    }
    .stat-cell:last-child { border-right: none; }
    .stat-num {
        font-family: 'Source Serif 4', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* ---------- Section cards ---------- */
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.6rem 1.8rem 0.4rem 1.8rem;
        margin-bottom: 1.4rem;
    }
    .section-head {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 1.1rem;
    }
    .section-index {
        font-family: 'Source Serif 4', serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--accent);
    }
    .section-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Widget label tightening */
    .stSlider label, .stSelectbox label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--ink-soft) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: var(--muted); }

    /* ---------- Button ---------- */
    .stButton button {
        background: var(--ink);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0.7rem 1.2rem;
        letter-spacing: 0.01em;
        transition: background 0.15s ease;
    }
    .stButton button:hover { background: var(--accent); color: white; }

    /* ---------- Result ---------- */
    .result-banner {
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin: 0.4rem 0 1.2rem 0;
        border-left: 3px solid;
    }
    .result-banner .result-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
        margin-bottom: 0.25rem;
    }
    .result-banner .result-value {
        font-family: 'Source Serif 4', serif;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .result-high  { background: #F2F9F5; color: #1F7A4D; border-color: #1F7A4D; }
    .result-avg   { background: #FEFAF2; color: #92650F; border-color: #C9860F; }
    .result-risk  { background: #FDF4F3; color: #A83A31; border-color: #C4453A; }

    .footer-note {
        color: var(--muted);
        font-size: 0.78rem;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-wrap">
    <div class="kicker">Machine Learning · Academic Analytics</div>
    <div class="doc-title">Student Performance Prediction System</div>
    <div class="doc-subtitle">
        Enter a student's profile once and receive a combined prediction drawn from
        three independently trained models — GPA regression, engagement classification,
        and grade-band classification.
    </div>
    <div class="stat-row">
        <div class="stat-cell">
            <div class="stat-num">3</div>
            <div class="stat-label">Trained Models</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">20+</div>
            <div class="stat-label">Input Features</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">93%</div>
            <div class="stat-label">Peak Model R²</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">Live</div>
            <div class="stat-label">Prediction</div>
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
# Section 1 — Basic Information
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-head"><span class="section-index">01</span><span class="section-title">Basic Information</span></div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 15, 22, 17)
    attendance_pct = st.slider("Attendance (%)", 0, 100, 90)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    absence_pct = 100 - attendance_pct
    st.metric("Absence rate", f"{absence_pct}%")

st.markdown("<div style='height:0.4rem'></div></div>", unsafe_allow_html=True)

absences = round((absence_pct / 100) * 30)
if attendance_pct >= 90:
    attend_habit = "Always"
elif attendance_pct >= 60:
    attend_habit = "Sometimes"
else:
    attend_habit = "Never"

# ---------------------------------------------------------
# Section 2 — Academics
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-head"><span class="section-index">02</span><span class="section-title">Academics</span></div>
""", unsafe_allow_html=True)

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

st.markdown("<div style='height:0.4rem'></div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 3 — Engagement
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-head"><span class="section-index">03</span><span class="section-title">Engagement</span></div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    raised_hands = st.slider("Hand raises / participation", 0, 100, 40)
    visited_resources = st.slider("Resource views", 0, 100, 50)
with col2:
    discussion = st.slider("Discussion participation", 0, 100, 30)
    youtube_learning = st.slider("Self-study hours (video/online)", 0, 20, 3)

st.markdown("<div style='height:0.4rem'></div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 4 — Activities
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-head"><span class="section-index">04</span><span class="section-title">Activities</span></div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    extracurricular = st.selectbox("Extracurricular activities?", ["No", "Yes"])
    sports = st.selectbox("Plays sports?", ["No", "Yes"])
with col2:
    music = st.selectbox("Music / dance?", ["No", "Yes"])
    volunteering = st.selectbox("Volunteering?", ["No", "Yes"])
    internet = st.selectbox("Internet at home?", ["Yes", "No"])

st.markdown("<div style='height:0.4rem'></div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
run_clicked = st.button("Get combined prediction", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running models..."):
        effective_study_hours = min(25, study_hours + round(youtube_learning * 0.5))

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
        predicted_cgpa = (predicted_gpa / 4.0) * 10.0

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

    students.insert_one({
        "age": age,
        "gender": gender,
        "attendance": attendance_pct,
        "study_hours": study_hours,
        "created_at": datetime.now()
    })

    predictions.insert_one({
        "predicted_gpa": float(predicted_gpa),
        "predicted_cgpa": float(predicted_cgpa),
        "behavior": behavior_level,
        "grade_band": grade_band,
        "overall": overall,
        "created_at": datetime.now()
    })

    st.markdown(
        f'''<div class="result-banner {css_class}">
                <div class="result-label">Overall Assessment</div>
                <div class="result-value">{overall}</div>
            </div>''',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted GPA", f"{predicted_gpa:.2f} / 4.0", f"{predicted_cgpa:.2f} CGPA")
    c2.metric("Engagement", behavior_level)
    c3.metric("Grade band", grade_band)

    st.caption(
        "Combined from three independently trained models — GPA regression, "
        "engagement classification, and grade-band classification."
    )

st.markdown(
    '<div class="footer-note">Built with scikit-learn &amp; Streamlit · Trained on three real-world student performance datasets</div>',
    unsafe_allow_html=True,
)