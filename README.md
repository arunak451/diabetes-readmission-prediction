# 🏥 Diabetes Hospital Readmission Prediction

Predicts whether a diabetic patient will be readmitted to hospital, using 100,000+ clinical encounter records. Identifying high-risk patients early helps hospitals plan follow-up care and reduce readmissions.

## Dataset

Diabetes 130-US hospitals dataset (`diabetic_data.csv`): demographics, admission details, lab procedures, medications and diagnoses per hospital encounter.

## Approach

1. Data cleaning: handling `?` placeholders and sparse columns
2. EDA and feature engineering on admissions, medications and diagnoses
3. Label encoding and scaling
4. Baseline models: **Random Forest** and **XGBoost**
5. Handling severe class imbalance (~11% of patients readmitted)
6. Hyperparameter tuning with **GridSearchCV**
7. Evaluation with confusion matrix, classification report, ROC curve and AUC
8. Interactive **Gradio** app for predictions; model saved with `pickle`

## Results

| Model | Readmission recall | ROC AUC |
|-------|--------------------|---------|
| Random Forest (baseline) | 0.00 | – |
| **XGBoost (tuned)** | **0.61** | **0.68** |

The baseline models reached 89% accuracy only by predicting "not readmitted" for almost everyone. The tuned XGBoost model trades raw accuracy for catching **61% of actual readmissions**, the outcome that matters clinically.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Gradio

## Run

```bash
git clone https://github.com/arunak451/diabetes-readmission-prediction.git
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn gradio
jupyter notebook "HOSPITAL DIABETES READMISSON PRED.ipynb"
```

The notebook expects `diabetic_data.csv` (originally run on Google Colab from `/content/`).
