# 🫁 COPD Severity Dashboard

An interactive, multi-tab data dashboard for exploring the global burden of Chronic Obstructive Pulmonary Disease (COPD) - from WHO population-level mortality trends down to individual patient-level predictions using machine learning.

Built with **Python Dash**, **Plotly**, and **scikit-learn / XGBoost**.

---

## 📖 Overview

This dashboard tells the full story of COPD across five interconnected sections:

1. **The COPD Story** - Introduction and narrative context
2. **Global Impact** - WHO mortality trends by region and year
3. **Patient Insights** - Lung function, exercise capacity, and symptom burden
4. **Demographics** - Age, gender, and severity interaction patterns
5. **Predictive Models** - ML-powered COPD severity prediction and model evaluation

---

## ✨ Features

- 🌍 **Global mortality trends** — Interactive line charts filtered by WHO region and year range
- 🔬 **Patient-level analysis** — Scatter plots, histograms, box plots, and heatmaps across severity levels
- 🤖 **Live severity prediction** — Enter patient vitals and get a predicted COPD severity (MILD / MODERATE / SEVERE / VERY SEVERE)
- 📊 **Model comparison** — Accuracy bar chart, ROC-AUC curves, and Precision-Recall curves for 5 ML models
- 📈 **Feature importance** — Linear regression coefficients for FEV1 and 6-Minute Walk Test predictors
- 🧭 **Journey progress bar** — Tracks your navigation through the dashboard

---

## 🤖 ML Models Included

| Model | Description |
|---|---|
| XGBoost | Gradient boosted trees (default/best performer) |
| Random Forest | Ensemble of decision trees |
| Logistic Regression | Baseline linear classifier |
| Naive Bayes | Probabilistic classifier |
| SVM | Support Vector Machine |

All models predict one of four COPD severity classes: `MILD`, `MODERATE`, `SEVERE`, `VERY SEVERE`.

---

## 🗂️ Project Structure

```
copd-severity-dashboard/
├── app.py                          # Main Dash application
├── requirements.txt
└── data/
    ├── dataset.csv                               # Patient-level COPD dataset
    ├── WHO.csv                                   # WHO global mortality data
    ├── model_comparison_results.csv              # Model comparison metrics
    ├── COPD_Model_Accuracies__Excluding_ANN_.csv # Per-model accuracy scores
    ├── roc_curve_data.json                       # Pre-computed ROC curve data
    ├── model_predictions.json                    # Saved predictions for PR curves
    ├── copd_xgboost.joblib                       # Trained XGBoost model
    ├── copd_random_forest.joblib                 # Trained Random Forest model
    ├── copd_naive_bayes.joblib                   # Trained Naive Bayes model
    ├── copd_logistic_regression.joblib           # Trained Logistic Regression model
    ├── copd_svm.joblib                           # Trained SVM model
    └── copd_severity_predictor_v1.0_best.joblib  # Preprocessing pipeline (imputer, scaler, encoder)
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AthulJkmr/copd-severity-dashboard.git
cd copd-severity-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Then open your browser and go to:
http://localhost:8050
```
---
| Live Rendered Link |
|---|
https://copd-severity-dashboard.onrender.com
---
## 📦 Dependencies

```
dash
dash-bootstrap-components
pandas
plotly
scikit-learn
xgboost
numpy
joblib
```

---

## 📊 Data Sources

| Dataset | Source |
|---|---|
| Patient COPD clinical data | [Kaggle](https://www.kaggle.com/) |
| Global COPD mortality | [WHO Global Health Observatory](https://www.who.int/data/gho) |

---

## 🩺 Clinical Features Used for Prediction

| Feature | Description |
|---|---|
| AGE | Patient age in years |
| PackHistory | Smoking history (pack-years) |
| FEV1 | Forced Expiratory Volume in 1 second (L) |
| FEV1PRED | FEV1 as % of predicted normal |
| FVC | Forced Vital Capacity (L) |
| FVCPRED | FVC as % of predicted normal |
| MWT1 / MWT2 | 6-Minute Walk Test distances (m) |
| CAT | COPD Assessment Test score (0–40) |
| HAD | Hospital Anxiety & Depression Scale |
| SGRQ | St. George's Respiratory Questionnaire score |
| gender, smoking, Diabetes, hypertension, AtrialFib, IHD, muscular | Demographics & comorbidities |

---

## 📸 Screenshots

> <img width="1280" height="708" alt="Screenshot 2026-03-25 at 2 05 21 am" src="https://github.com/user-attachments/assets/1d2809c9-6281-4b1f-b219-a04f2c6697fb" />
> <img width="1280" height="713" alt="image" src="https://github.com/user-attachments/assets/8cf24f6c-f808-4df9-b1e8-abc451e5fa20" />
> <img width="1280" height="706" alt="image" src="https://github.com/user-attachments/assets/d0d6cf03-4c58-431f-88ff-d8f42d7dabfa" />
> <img width="1280" height="709" alt="image" src="https://github.com/user-attachments/assets/fb2f1c94-190d-4f01-8c66-0da9a3e0caee" />
> <img width="1280" height="666" alt="image" src="https://github.com/user-attachments/assets/334d6205-0126-4915-901d-8539fb09c8f0" />
---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- WHO Global Health Observatory for mortality data
- Kaggle community for the patient dataset
- Plotly / Dash for the interactive visualisation framework

---

## 📇 Contact Me
| Email | athul27ks@gmail.com |
|---|---|
| LinkedIn | www.linkedin.com/in/athul27ks |
