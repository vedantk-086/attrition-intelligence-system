# Workforce Retention Intelligence System

Dataset: [IBM HR Analytics Employee Attrition & Performance (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

An XGBoost model that predicts whether an employee is likely to leave a company, built on IBM's HR dataset. The focus here wasn't just getting a high accuracy number — the dataset is imbalanced (only ~16% of employees actually leave), so a model that just predicts "stay" for everyone would already look 84% accurate while being completely useless. This project handles that properly.

## The problem with a naive approach

At the default decision threshold, the model hit 85%+ accuracy but only caught 21% of the employees who actually left — it was basically guessing "stay" most of the time and getting away with it because of the imbalance. That's not something HR can act on.

To fix this, I used `scale_pos_weight` in XGBoost to rebalance the training, then tested different probability thresholds instead of sticking with the default 0.5. Threshold 0.3 gave the best trade-off — accuracy dropped a bit, but recall nearly doubled.

## Workflow

Data Cleaning → Feature Selection (correlation-based) → Feature Engineering → Encoding → Train-Test Split → XGBoost Training → Class Imbalance Handling → Threshold Tuning → Evaluation → Streamlit App

## Engineered Features

- `IncomePerYearAtCompany` — income relative to tenure
- `HighOvertimeRisk` — flags employees doing overtime
- `TravelStress` — combines travel frequency with commute distance
- `JobHopperRisk` — low satisfaction + low involvement together
- `LowIncomeHighDissatisfaction` — underpaid and dissatisfied employees

`HighOvertimeRisk` ended up being the strongest predictor in the model, ahead of every raw feature.

## Results

| Metric | Score |
|---|---|
| Accuracy | 81.6% |
| Precision | 43.1% |
| Recall | 46.8% |
| F1 Score | 0.449 |
| ROC-AUC | 0.742 |

## Demo

**High-risk profile** (young employee, low income, frequent travel, overtime, low satisfaction):
> Prediction: Likely to Leave — 98.9% probability

**Low-risk profile** (senior employee, high income, no overtime, high satisfaction):
> Prediction: Likely to Stay — 0.1% probability

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## Project Structure

```
workforce-retention/
├── train_model.py            # Preprocessing + model training
├── app.py                    # Streamlit app
├── requirements.txt
├── encoders.pkl
├── xgb_attrition_model.json
└── README.md
```

## Run It Yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```
