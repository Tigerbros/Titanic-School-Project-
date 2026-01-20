# 🚢 Titanic Survival Prediction - Streamlit App

## Course: CSC334 - Python Programming Language II

---

## 📋 Project Overview

A web application that predicts whether a passenger would have survived the Titanic disaster based on their characteristics (class, sex, age, fare, embarkation port).

---

## 🏗️ Project Structure

```
titanic-survival-app/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── requirements.txt                # Version-pinned dependencies
├── model/
│   └── titanic_survival_model.pkl  # Trained pipeline (generated)
└── README.md                       # This file
```

---

## 🔧 Technical Approach

### ML Pipeline (Industry Standard)

The model uses a **scikit-learn Pipeline** that combines:

1. **Preprocessing** (ColumnTransformer)
   - Numeric features: Median imputation → StandardScaler
   - Categorical features: Most frequent imputation → OneHotEncoder

2. **Classifier**: Logistic Regression

### Why This Approach?

- ✅ No data leakage (fit on training data only)
- ✅ Single `.pkl` file contains everything
- ✅ No manual preprocessing in Streamlit app
- ✅ Handles missing values automatically
- ✅ Industry-standard workflow

---

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

This creates `model/titanic_survival_model.pkl`

### Step 3: Run Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Deploy!

**Note:** Make sure `model/titanic_survival_model.pkl` is committed to the repo.

---

## 📊 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Pclass | Numeric | Passenger class (1, 2, 3) |
| Sex | Categorical | Gender (male, female) |
| Age | Numeric | Age in years |
| Fare | Numeric | Ticket fare (£) |
| Embarked | Categorical | Port (S, C, Q) |

---

## ✅ Marking Scheme Checklist

- [x] Version-pinned requirements.txt
- [x] Proper preprocessing pipeline
- [x] Missing value strategy (median/mode)
- [x] StandardScaler correctly applied
- [x] No data leakage
- [x] Single model file deployment
- [x] Clean Streamlit integration
- [x] User-friendly interface

---

## 👨‍💻 Author

Machine Learning Engineer

**Date:** January 2026
