# Breast Cancer Diagnosis API

This is a FastAPI-based backend for predicting breast tumor malignancy using machine learning and neural network models. It powers real-time classification via a simple REST API.

## Live API

> Coming soon (e.g. Render URL)

## Models Included

- Logistic Regression (with 5 selected features)
- Random Forest (full 30 features)
- Neural Network (TensorFlow, full 30 features)

All models are pre-trained and loaded from disk using `joblib` and `tensorflow.keras`.

## Endpoints

### `/predict/randomforest`
- Input: 30 features
- Output: `0` (Benign) or `1` (Malignant)

### `/predict/logistic`
- Input: 5 selected features
  - `radius_mean`
  - `texture_mean`
  - `compactness_worst`
  - `concave_points_worst`
  - `area_worst`
- Output: `0` or `1`

### `/predict/neuralnet`
- Input: 30 features
- Output: `0` or `1` with probability
