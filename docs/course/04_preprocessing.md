# Module 04 — Preprocessing Pipelines

## Numeric features
Numeric variables are standardized using:

z = (x - mean) / standard_deviation

## Categorical features
One-hot encoding creates indicator columns for categories.

Example:
Contract = Month-to-month / One year / Two year

becomes binary encoded columns.

## Why use a Pipeline?
A Scikit-learn Pipeline keeps preprocessing and inference together.

Benefits:
- Less leakage risk
- Reproducible training
- Same transformations during prediction
- Easier serialization
- Cleaner production code

## Interview point
Always fit preprocessing on the training data, never independently on the complete dataset before the train/test split.
