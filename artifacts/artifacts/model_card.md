# Model Card — Wine Quality Prediction

## 1. Model Overview

This model predicts wine quality scores based on physicochemical attributes using an XGBoost Regressor optimized with AWS SageMaker Hyperparameter Optimization.

The project is designed as an end-to-end Machine Learning Engineering case study, covering data preparation, model training, hyperparameter tuning, evaluation and deployment readiness.

---

## 2. Intended Use

The model is intended to support analytical and operational decisions related to wine quality assessment.

Potential use cases include:

- preliminary quality screening;
- prioritization of wine batches for inspection;
- support for quality control teams;
- experimentation with machine learning workflows;
- demonstration of MLOps-ready model development.

This model should be used as a decision-support tool, not as a replacement for expert sensory evaluation.

---

## 3. Input Features

The model uses physicochemical properties of wine as input features.

Expected features include:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- wine type indicator

---

## 4. Target Variable

The target variable is the wine quality score.

This is treated as a regression problem, where the model estimates a continuous quality score based on the available physicochemical attributes.

---

## 5. Model Type

- Algorithm: XGBoost Regressor
- Optimization: AWS SageMaker Hyperparameter Optimization
- Objective metric: validation RMSE
- Problem type: supervised regression

---

## 6. Evaluation Metrics

The model was evaluated using regression metrics:

| Metric | Value |
|---|---:|
| MSE | 0.4564 |
| MAE | 0.3867 |
| RMSE | 0.6756 |

The main optimization metric is RMSE, because it penalizes larger errors while remaining interpretable in the same scale as the target variable.

---

## 7. Limitations

This model has important limitations:

- The dataset is public and may not represent all wine production contexts.
- The target variable is based on historical quality scores, which may include subjectivity.
- The model does not replace expert sensory analysis.
- External factors such as production region, storage conditions and grape variety may not be fully represented.
- Deployment in a real production environment would require additional monitoring, validation and governance.

---

## 8. Risks and Considerations

Potential risks include:

- over-reliance on model predictions without expert validation;
- performance degradation when applied to data from different production contexts;
- bias due to limited dataset scope;
- lack of monitoring if deployed without drift detection.

Recommended controls:

- monitor input data distributions;
- periodically re-evaluate model performance;
- validate predictions against expert assessments;
- document changes in data, model and hyperparameters.

---

## 9. Deployment Readiness

The model is currently structured as a portfolio-ready artifact and is prepared for API-based serving.

Planned deployment components include:

- FastAPI prediction endpoint;
- Docker containerization;
- local inference testing;
- optional cloud deployment;
- future monitoring layer.

---

## 10. Responsible Use

This model should be used to support quality analysis, not to automate final commercial or technical decisions without human review.

Any real-world use should include:

- validation with domain experts;
- monitoring of data drift;
- periodic retraining;
- clear documentation of assumptions and limitations.
