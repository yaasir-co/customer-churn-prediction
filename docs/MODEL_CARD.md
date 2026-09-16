# Model Card — Customer Churn Project

## Intended use
Educational demonstration, portfolio development and interview preparation.

## Dataset
Synthetic telecom-style dataset with 9,000 rows.

## Target
`churn` where 1 indicates churn and 0 indicates stay.

## Candidate models
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

## Selection
Best model by held-out ROC-AUC: **Logistic Regression**

## Limitations
- Synthetic data is not representative of any particular telecom company.
- Feature importance is predictive, not causal.
- Real production use requires organization-specific validation, privacy controls, monitoring and fairness review.
