# Streamlit Deployment Fix

This package is designed for browser-based GitHub upload.

## Important
All model files are under GitHub's browser upload limit.

The saved Scikit-learn models were produced with:

- scikit-learn: 1.8.0
- joblib: 1.5.3

`requirements.txt` pins the compatible Scikit-learn version.

## Upload
Make sure GitHub contains the entire `models` folder, especially:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/gradient_boosting.joblib`
- `models/knearest_neighbors.joblib`
- `models/model_metrics.json`
- `models/threshold_metrics.csv`
- `models/feature_importance.csv`

Then reboot/redeploy the Streamlit app.
