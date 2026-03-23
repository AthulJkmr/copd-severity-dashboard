# COPD Severity Prediction Dashboard

An interactive analytics dashboard for predicting and visualising COPD (Chronic Obstructive Pulmonary Disease) severity using machine learning, built with Python and Plotly Dash.

## Live Demo
https://copd-severity-dashboard.onrender.com

---

## Overview

This project combines clinical patient data with WHO global statistics to build an end-to-end analytics and prediction tool for COPD severity classification. It was developed as part of a Master of Data Science project at Swinburne University of Technology.

**COPD severity classes:** MILD · MODERATE · SEVERE · VERY SEVERE

---

## Model Performance

| Model | Test Accuracy |
|-------|--------------|
| XGBoost (primary) | **90.48%** |
| Random Forest (baseline) | 80.95% |

Evaluated on a stratified 80/20 train-test split of 101 patient records.

---

## Features

- **Severity Prediction** — input patient clinical indicators and get an instant ML-powered severity classification
- **Patient Analytics** — filter and explore the dataset by severity, gender, and age range
- **WHO Global Trends** — visualise worldwide COPD prevalence and mortality data
- **Model Comparison** — side-by-side XGBoost vs Random Forest accuracy display
- **Interactive Charts** — built with Plotly for hover, zoom, and drill-down

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Dashboard | Dash, Dash Bootstrap Components |
| ML Models | XGBoost, Scikit-learn (Random Forest, preprocessing pipelines) |
| Data | Pandas, NumPy |
| Visualisation | Plotly Express, Plotly Graph Objects |

---

## Project Structure

```
copd-severity-dashboard/
├── app.py
├── Procfile
├── requirements.txt
├── README.md
├── copd_severity_predictor_v1_0.joblib
├── .python-version
├── runtime.txt
└── data/
    ├── dataset.csv
    └── WHO.csv
```

---

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/copd-severity-dashboard.git
cd copd-severity-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open your browser at `http://127.0.0.1:8050`

---

## Dataset

The clinical dataset contains **101 patient records** across **24 features** including:

- Demographic: `AGE`, `gender`, `AGEquartiles`
- Pulmonary function: `FEV1`, `FEV1PRED`, `FVC`, `FVCPRED`
- Exercise capacity: `MWT1`, `MWT2`, `MWT1Best`
- Symptom scores: `CAT`, `HAD`, `SGRQ`
- Comorbidities: `Diabetes`, `hypertension`, `AtrialFib`, `IHD`, `muscular`
- Lifestyle: `PackHistory`, `smoking`
- Target: `COPDSEVERITY` (MILD / MODERATE / SEVERE / VERY SEVERE)

---

## Author

**Athul Jayakumar** · [LinkedIn](https://www.linkedin.com/in/athul27ks) · athul27ks@gmail.com

Master of Data Science, Swinburne University of Technology (2025)
