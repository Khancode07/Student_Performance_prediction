# 🎓 Student Performance Prediction System


---

## 📌 Ye Project Kya Karta Hai

Is project me **3 alag real student datasets** use kiye gaye hain, aur student performance ko 3 alag tariko se predict kiya jata hai:

1. **School GPA Prediction** — Student ka GPA (0.0 se 4.0 tak) predict karta hai — age, study time, absences, parental support, etc. ke basis pe.
2. **Behavior Classification** — Student ka performance level Low / Medium / High predict karta hai — class me kitna active hai (hand raise karna, resources dekhna, discussion me part lena) uske basis pe.
3. **University Grade Prediction** — University student ka grade band (Low / High) predict karta hai — study hours, attendance, scholarship, lifestyle factors ke basis pe.

Teeno models ko **ek simple dashboard** me combine kiya gaya hai — jahan tum ek hi baar student ki details bharo, aur ek final result milta hai.

---

## 🧠 Results

| Model | Kya Predict Karta Hai | Accuracy |
|---|---|---|
| GPA Prediction | GPA (number) | 92.83% (R² score) |
| Behavior Classification | Low / Medium / High | 77.08% |
| Grade Prediction | Low / High | 82.76% |

**Sabse bada insight:** Teeno datasets me, **attendance/absences/Behavior** hi sabse strong factor nikla jo student performance ko affect karta hai.

---

## 🧩 Machine Learning Yahan Kaise Kaam Karti Hai (Simple Explanation)

Rule likhne ke bajaye jaise *"agar absences > 10 hai to GPA low hoga,"* hum computer ko hazaron real student records dikhate hain jinme humein pehle se pata hai ki asli GPA/grade kya tha — aur computer khud pattern **seekh leta hai**. Isi ko **training** kehte hain. Training ke baad, hum computer ko ek naye student ki details dete hain (bina answer bataye) aur wo **predict** kar deta hai.

**Algorithm jo use kiya:** Teeno models me **Random Forest** use hua hai.
> Socho ki 300 alag "mini-experts" (jinhe decision trees kehte hain) se ek-ek guess maango, har ek thoda alag tarike se data dekhta hai, aur phir sabka average/majority answer le lo. Zyada opinions = ek akele guesser se zyada reliable, accurate answer.

**Is project me 2 tarah ki predictions hoti hain:**

| Type | Kya Predict Karta Hai | Kahan Use Hua |
|---|---|---|
| **Regression** | Ek number | Module 1 (GPA, jaise 2.74) |
| **Classification** | Ek category | Module 2 & 3 (Low/Medium/High) |

---

## 📗 Module 1 — GPA Prediction (Regression)

Student ka GPA predict karta hai 12 features se: age, study time, absences, parental support, tutoring, extracurriculars, etc. **2,392 real student records** pe train kiya gaya.

**Result: R² = 0.93**

**R² ka matlab kya hai:** R² batata hai ki model ki predictions kitni achi hain, 0 se 1 ke scale pe.
- R² = 1.0 → har baar perfect prediction (real life me kam hi hota hai)
- R² = 0.0 → model basically random guess kar raha hai
- **R² = 0.93** → matlab model 93% variation explain kar pa raha hai students ke GPA me, jo humne diye features se — ye bahut strong result hai. Baaki 7% un cheezon ki wajah se hai jo humne measure hi nahi ki (jaise natural ability, mental health, etc.)

Iske alawa **MAE** (Mean Absolute Error — prediction average kitna off hai; MAE = 0.19 matlab predictions average 0.19 GPA points se off hain — bahut close) aur **RMSE** ([isi] jaisa, lekin badi galtiyon ko zyada punish karta hai) bhi dikhte hain.

**Sabse important factor:** Absences — model ke decision me 86% importance rakhta hai. Jitni zyada class miss hoti hai, GPA utna hi kam predict hota hai.

---

## 📙 Module 2 — Behavior Classification

Student ka performance **Low / Medium / High** predict karta hai — sirf classroom engagement ke basis pe — kitni baar hand raise kiya, course resources dekhe, announcements dekhe, discussions me part liya. **480 real student records** pe train kiya gaya.

**Result: Accuracy = 77.08%**

**Accuracy ka matlab kya hai:** Simple hai — 100 naye students test karo to model unme se 77 baar sahi performance level guess karega. 100% isliye nahi kyunki real life me ek jaisa behavior rakhne wale students ke bhi different outcomes ho sakte hain, un factors ki wajah se jo humne measure nahi kiye. 3-category prediction ke liye 77% ek solid, realistic result hai.

**Sabse important factors:** Student kitni baar online resources dekhta hai, aur kitni baar class me hand raise karta hai.

---

## 📘 Module 3 — University Grade Prediction

University student ka overall grade band predict karta hai — **Low ya High** — lifestyle aur study habits se (study hours, attendance, scholarship, part-time work, parents ki education). **145 real student records** pe train kiya gaya (baaki dono se kaafi chota dataset).

**Sirf 2 categories kyun (3 ke bajaye)?** Original data me 8 fine-grained grade levels the, lekin sirf 145 students hone ki wajah se kuch levels me sirf 2-3 students the — itna kam ki model reliably seekh hi nahi sakta. Isliye 2 broader bands (Low/High) me group kar diya, taaki task learnable ho jaye.

**Result: Accuracy = 82.76%**

**SMOTE kya hai (aur maine kyun use kiya):** Dataset me "Low" students, "High" students se zyada the (imbalanced data). Jab model imbalanced data pe train hota hai, to wo lazy ho jata hai aur zyadatar bade category ka hi guess karta rehta hai. **SMOTE (Synthetic Minority Oversampling Technique)** isko fix karta hai — chhoti category ke naye, realistic synthetic examples banata hai, real data ke patterns ke basis pe — isse training data balance ho jata hai aur model dono categories ko fairly seekhta hai. SMOTE use karne se is model ki accuracy kaafi improve hui.

---

## 🔗 Combined Dashboard Kaise Kaam Karta Hai

Kyunki teeno models alag *type* ka result dete hain (ek number vs. do alag category systems), inhe seedha average nahi kar sakte. Dashboard ye karta hai:

1. Predicted GPA ko score me convert karta hai: 0 (Low), 1 (Medium), ya 2 (High).
2. Engagement Level (jo already Low/Medium/High hai) ko bhi usi 0/1/2 scale me convert karta hai.
3. Grade Band (Low/High) ko bhi usi 0/1/2 scale me convert karta hai.
4. Teeno scores ko **average** karta hai.
5. Ek final verdict dikhata hai: **"At Risk"**, **"Average Performer"**, ya **"High Performer"** — saath me individual GPA, Engagement Level, aur Grade Band results bhi.

Isse teeno alag-alag trained models, teeno alag datasets pe, phir bhi ek meaningful combined answer de pate hain.

---

## 💻 Project Ke Andar Kya Hai

student-performance-system/
├── module1_gpa_prediction/            → GPA prediction model + data
├── module2_behavior_classification/   → Behavior classification model + data
├── module3_gpa_trend/                 → University grade prediction model + data
├── predict.py                         → Command-line version (terminal me answers type karo)
├── app.py                             → Web dashboard (Streamlit)
└── README.md  and readme_hindi.md     → Ye file

---

## ⚙️ Tools Use Kiye

- **Python**
- **pandas** — data cleaning ke liye
- **scikit-learn** — ML models banane ke liye (Random Forest)
- **imbalanced-learn (SMOTE)** — chhote university dataset ko balance karne ke liye
- **Streamlit** — web dashboard ke liye

---

## 🚀 Project Kaise Run Karein

### Step 1 — Zaroori tools install karo
pip install pandas numpy scikit-learn joblib streamlit imbalanced-learn

### Step 2 — Models train karo
Har module folder ke andar jaake uska training script run karo:

cd module1_gpa_prediction
python train_model.py

cd ../module2_behavior_classification
python train_model.py

cd ../module3_gpa_trend
python train_model.py

Isse har folder me trained model files (.pkl) ban jayengi.

### Step 3 — Web dashboard run karo
cd ..
streamlit run app.py

Ye browser me http://localhost:8501 pe dashboard khol dega.

---

## 📊 Dashboard Kaise Use Karein

1. Ek form bharo — study hours, attendance %, class participation, extracurriculars, learning habits, etc.
2. "Get Combined Prediction" button dabao.
3. Teeno models background me ek saath chalte hain.
4. Ek final result milta hai: Predicted GPA + Engagement Level + Grade Band + ek overall verdict.

---

## 🔍 Ye Project Banate Waqt Kya Seekha

- Real-world datasets aksar ek doosre se match nahi karte — mujhe teeno alag models train karne pade, ek me merge karne ke bajaye, kyunki unke features me bahut kam overlap tha.
- Chhote datasets (jaise 145-row university wala) ko simpler target categories aur SMOTE jaisi techniques chahiye hoti hain, taaki classes balance ho aur reliable results milein.
- Attendance aur active class participation har dataset me performance ke sabse strong predictors nikle.
- Multiple independently trained models ko ek dashboard me combine karne ke liye, har model ke alag output format ko ek common scale me convert karna padta hai, tabhi unhe ek overall verdict me jod sakte hain.

---

---

## 🧪 Ek Experiment Jo Kaam Nahi Aaya (Aur Kyun)

Module 1 banate waqt, maine ek alag approach bhi try ki thi — UCI Student Performance dataset (Math + Portuguese course records) use karke, sirf lifestyle factors se final grade predict karna, bina pichle grades ka data use kiye. Isse R² sirf 0.25 (25%) aaya, kyunki bina past performance ki koi info ke final grade predict karna genuinely mushkil task hai.

Maine original, strong wali approach hi final Module 1 model ke liye rakhi (GPA predict karna demographic, study, aur behavioral features se — R² = 0.93). Ye experiment ek achha example hai ki same modeling technique alag-alag results de sakti hai, depending on kaunse features available hain aur underlying prediction task kitna "hard" hai.



## 📚 Quick Glossary

| Term | Simple Matlab |
|---|---|
| Model | Wo trained "brain" jo predictions karta hai |
| Training | Model ko data se sikhana jahan answer pehle se pata ho |
| Feature | Ek input jo model use karta hai (jaise Age, Absences, Study Hours) |
| Target | Jo hum predict karna chahte hain (jaise GPA, Grade) |
| Regression | Ek number predict karna |
| Classification | Ek category predict karna |
| Random Forest | Ek algorithm jo kai decision trees combine karke reliable answer deta hai |
| R² | Regression model data ko kitna achi tarah explain karta hai (0 se 1, jitna zyada utna better) |
| Accuracy | Classification model ki % correct predictions |
| MAE / RMSE | Prediction errors ka average size (kam hona better hai) |
| SMOTE | Ek technique jo dataset balance karti hai jab ek category ke examples doosri se bahut kam ho |
| Feature Importance | Model ke decision me kaunse inputs sabse zyada matter karte hain |
| Train/Test Split | Data ko do parts me split karna — ek model ko sikhane ke liye, ek check karne ke liye ki sach me sikha ya nahi |

---

## 👤 Author

Ek khan ka project ke roop me banaya gaya — data cleaning, machine learning, aur ek real interactive dashboard banane ki practice karne ke liye.