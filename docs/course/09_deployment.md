# Module 09 — Deployment

## Demo deployment
Streamlit provides UI and inference in one application.

## Production deployment
A larger architecture normally separates:
- UI
- Prediction API
- Feature processing
- Model artifact
- Logging
- Monitoring
- Data storage

## API lifecycle
Request -> validate -> preprocess -> predict probability -> apply policy -> return response -> log metadata

## Production concerns
- Authentication
- Input validation
- Latency
- Model version
- Dependency versions
- Failure handling
- Privacy
