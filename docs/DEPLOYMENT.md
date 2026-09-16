# Deployment Guide

## Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload all files in this project.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app.
5. Select your repository and `main` branch.
6. Set the main file to `app.py`.
7. Deploy.

## Production evolution
For a commercial production system, separate the model service from the UI:
- FastAPI model API
- Authentication
- PostgreSQL / warehouse
- Logging and monitoring
- Model registry
- CI/CD
- Drift detection
- Scheduled retraining
