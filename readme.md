# 🎓 Student Performance Prediction System

A simple machine learning project that predicts how well a student will perform, based on things like study hours, attendance, and classroom engagement.

---

## 📌 What This Project Does

This project looks at **3 different real student datasets** and predicts student performance in 3 different ways:

1. **School GPA Prediction** — Predicts a student's GPA (0.0 to 4.0) based on age, study time, absences, parental support, etc.
2. **Behavior Classification** — Predicts if a student's performance is Low / Medium / High, based on how active they are in class (raised hands, resources visited, discussions).
3. **University Grade Prediction** — Predicts a university student's grade band (Low / High) based on study hours, attendance, scholarship, and lifestyle factors.

All 3 models are combined into **one simple dashboard** where you enter a student's details once, and get one final result.

---

## 🧠 Results

| Model | What it Predicts | Accuracy |
|---|---|---|
| GPA Prediction | GPA (number) | 92.83% (R² score) |
| Behavior Classification | Low / Medium / High | 77.08% |
| Grade Prediction | Low / High | 82.76% |

**Biggest finding:** Across all 3 datasets, **attendance/absences** turned out to be one of the strongest factors affecting student performance.

---

## 🧩 How Machine Learning Works Here (Simple Explanation)

Instead of writing rules like *"if absences > 10, GPA is low,"* we show the computer thousands of real student records where we already know the actual GPA/grade, and the computer **learns the pattern itself**. This is called **training**. Once trained, we can give it a brand-new student's details (without the answer) and it will **predict** the result.

**The algorithm used:** All 3 models use a **Random Forest**.
> Imagine asking 300 different "mini-experts" (called decision trees) to each make their own guess by looking at the data slightly differently, then taking the average/majority answer. More opinions = a more reliable final answer than trusting just one guesser.

**Two types of predictions in this project:**

| Type | What it predicts | Used in |
|---|---|---|
| **Regression** | A number | Module 1 (GPA, e.g. 2.74) |
| **Classification** | A category | Module 2 & 3 (Low/Medium/High) |

---

## 📗 Module 1 — GPA Prediction (Regression)

Predicts a student's GPA using 12 features: age, study time, absences, parental support, tutoring, extracurriculars, etc. Trained on **2,392 real student records**.

**Result: R² = 0.93**

**What R² means:** R² tells you how good the model's predictions are, on a scale from 0 to 1.
- R² = 1.0 → perfect predictions every time (rare in real life)
- R² = 0.0 → the model is basically guessing randomly
- **R² = 0.93** → the model explains 93% of the differences in students' GPAs using the features we gave it — a very strong result. The remaining 7% comes from things we didn't measure (natural ability, mental health, etc.)

You'll also see **MAE** (Mean Absolute Error — average size of the prediction mistake; MAE = 0.19 means predictions are off by about 0.19 GPA points on average) and **RMSE** (similar, but punishes big mistakes more).

**Most important factor:** Absences — responsible for 86% of the model's decision-making. The more classes missed, the lower the predicted GPA.

---

## 📙 Module 2 — Behavior Classification

Predicts if a student's performance is **Low / Medium / High**, based purely on classroom engagement — how often they raised their hand, visited course resources, viewed announcements, and joined discussions. Trained on **480 real student records**.

**Result: Accuracy = 77.08%**

**What Accuracy means:** Out of every 100 new students tested, the model correctly guesses their performance level about 77 times. It's not 100% because real students with similar behavior can still have different outcomes due to factors we didn't measure — 77% is a solid, realistic result for a 3-category prediction.

**Most important factors:** How often the student visited online resources, and how often they raised their hand in class.

---

## 📘 Module 3 — University Grade Prediction

Predicts a university student's overall grade band — **Low or High** — from lifestyle and study habits (study hours, attendance, scholarship, part-time work, parents' education). Trained on **145 real student records**.

**Why only 2 categories instead of 3?** The original data had 8 fine-grained grade levels, but with only 145 students total, some levels had just 2-3 students — too few to learn from reliably. Grouping into 2 broader bands (Low/High) made the task learnable.

**Result: Accuracy = 82.76%**

**What is SMOTE (and why I used it):** The dataset had more "Low" students than "High" students (imbalanced data). A model trained on imbalanced data tends to get lazy and just guess the bigger category most of the time. **SMOTE (Synthetic Minority Oversampling Technique)** fixes this by generating new, realistic synthetic examples of the smaller category, based on patterns in the real data — balancing the training set so the model learns both categories fairly. Using SMOTE improved this model's accuracy noticeably.

---

## 🔗 How the Combined Dashboard Works

Since the 3 models output different *types* of results (a number vs. two different category systems), they can't just be averaged directly. The dashboard:

1. Converts the predicted GPA into a score: 0 (Low), 1 (Medium), or 2 (High).
2. Converts the Engagement Level (already Low/Medium/High) into the same 0/1/2 scale.
3. Converts the Grade Band (Low/High) into the same 0/1/2 scale.
4. **Averages** all three scores.
5. Shows one final verdict: **"At Risk"**, **"Average Performer"**, or **"High Performer"** — alongside the individual GPA, Engagement Level, and Grade Band results.

This lets three completely separately trained models, on three completely different datasets, still produce one meaningful combined answer.

---

## 💻 What's Inside This Project

student-performance-system/
├── module1_gpa_prediction/            → GPA prediction model + data
├── module2_behavior_classification/   → Behavior classification model + data
├── module3_gpa_trend/                 → University grade prediction model + data
├── predict.py                         → Command-line version (type answers in terminal)
├── app.py                             → Web dashboard (Streamlit)
└── README.md                          → This file

---

## ⚙️ Tools Used

- **Python**
- **pandas** — for cleaning data
- **scikit-learn** — for building the ML models (Random Forest)
- **imbalanced-learn (SMOTE)** — for balancing the small university dataset
- **Streamlit** — for the web dashboard

---

## 🚀 How to Run This Project

### Step 1 — Install the required tools
pip install pandas numpy scikit-learn joblib streamlit imbalanced-learn

### Step 2 — Train the models
Go inside each module folder and run its training script:

cd module1_gpa_prediction
python train_model.py

cd ../module2_behavior_classification
python train_model.py

cd ../module3_gpa_trend
python train_model.py

This creates the trained model files (.pkl) inside each folder.

### Step 3 — Run the web dashboard
cd ..
streamlit run app.py

This opens the dashboard in your browser at http://localhost:8501.

---

## 📊 How to Use the Dashboard

1. Fill in one form — study hours, attendance %, class participation, extracurriculars, learning habits, etc.
2. Click "Get Combined Prediction".
3. All 3 models run in the background at once.
4. Get one final result: Predicted GPA + Engagement Level + Grade Band + an overall verdict.

---

## 🔍 What I Learned Building This

- Real-world datasets often don't match each other — I had to train 3 separate models instead of merging everything into one, since their features barely overlap.
- Small datasets (like the 145-row university one) need simpler target categories and techniques like SMOTE to balance classes and get reliable results.
- Attendance and active class participation were the strongest predictors of performance across every dataset I tested.
- Combining multiple independently trained models into a single dashboard requires converting each model's different output format into a common scale before they can be combined into one overall verdict.


---

## 🧪 An Experiment That Didn't Work (and Why)

While building Module 1, I also tried an alternate approach using the UCI Student Performance dataset (Math + Portuguese course records), predicting final grades from lifestyle factors alone — without using any earlier grade data. This resulted in a much lower R² of only 0.25 (25%), since predicting a final grade with zero information about past performance is a genuinely hard task.

I kept the original, stronger approach (predicting GPA from demographic, study, and behavioral features — R² = 0.93) as the final Module 1 model instead. This experiment is a good example of how the same modeling technique can perform very differently depending on which features are available and how "hard" the underlying prediction task really is.


---

## 📚 Quick Glossary

| Term | Simple Meaning |
|---|---|
| Model | The trained "brain" that makes predictions |
| Training | Teaching the model using data where the answer is already known |
| Feature | An input the model uses (e.g., Age, Absences, Study Hours) |
| Target | What we're trying to predict (e.g., GPA, Grade) |
| Regression | Predicting a number |
| Classification | Predicting a category |
| Random Forest | An algorithm combining many decision trees for a more reliable answer |
| R² | How well a regression model explains the data (0 to 1, higher = better) |
| Accuracy | % of correct predictions for a classification model |
| MAE / RMSE | Average size of prediction errors (lower = better) |
| SMOTE | A technique to balance a dataset when one category has far fewer examples than another |
| Feature Importance | Which inputs mattered most to the model's decision |
| Train/Test Split | Splitting data into a part to teach the model and a part to check if it actually learned well |

---
# What i Learned
## 👤 Author

Built as a khan project to practice data cleaning, machine learning, and building a real interactive dashboard.