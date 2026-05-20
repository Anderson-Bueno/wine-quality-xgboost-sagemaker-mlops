# Wine Quality Prediction with XGBoost and SageMaker HPO

**End-to-end Machine Learning Engineering project for wine quality prediction using XGBoost, AWS SageMaker Hyperparameter Optimization, model evaluation, artifact tracking and API deployment with FastAPI + Docker.**

---

## 🔎 Overview

This project presents an end-to-end Machine Learning Engineering workflow for predicting wine quality based on physicochemical properties.

The goal is not only to train a model, but to structure a reproducible and production-oriented machine learning pipeline covering:

- data ingestion;
- exploratory data analysis;
- data preprocessing;
- train/validation/test split;
- model training with XGBoost;
- hyperparameter optimization with AWS SageMaker;
- model evaluation;
- artifact management;
- model documentation;
- API deployment with FastAPI;
- Docker-based execution;
- technical documentation for portfolio and LinkedIn publication.

This project was designed as a practical case study to demonstrate the transition from experimentation to a more production-ready ML workflow.

---

## 📌 Business Problem

Wine quality assessment is traditionally based on sensory evaluation, which can be subjective, costly and difficult to scale.

Using physicochemical attributes, machine learning models can support quality control teams by estimating wine quality scores and helping prioritize batches for further inspection.

### Central question

> How can we build a reproducible ML pipeline to predict wine quality, optimize model performance and prepare the solution for deployment?

---

## 🎯 Objectives

The main objectives of this project are:

- combine red and white wine datasets;
- explore correlations and data distributions;
- detect and analyze outliers;
- prepare data for supervised learning;
- train an XGBoost regression model;
- optimize hyperparameters using SageMaker HPO;
- evaluate the final model using regression metrics;
- store model metrics and hyperparameters as artifacts;
- document the model through a model card;
- expose predictions through a FastAPI endpoint;
- containerize the API with Docker;
- prepare a concise LinkedIn publication linked to the full GitHub project.

---

## 🏗️ Solution Architecture

```text
data/raw
   ↓
data preprocessing
   ↓
EDA and feature analysis
   ↓
train / validation / test split
   ↓
S3 upload
   ↓
SageMaker XGBoost training
   ↓
Hyperparameter Optimization
   ↓
Best model extraction
   ↓
Model evaluation
   ↓
Artifact generation
   ↓
FastAPI prediction service
   ↓
Docker deployment
```

---

## 📁 Project Structure

```text
.
├── README.md
├── requirements.txt
├── Dockerfile
│
├── data/
│   ├── raw/
│   │   ├── dataset_1.csv
│   │   └── dataset_2.csv
│   └── processed/
│
├── notebooks/
│   └── Projeto2.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_local.py
│   ├── evaluate.py
│   └── utils.py
│
├── sagemaker/
│   ├── hpo_config.json
│   ├── train_sagemaker.py
│   └── best_hyperparameters.json
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── models/
│   └── xgboost-model
│
├── artifacts/
│   ├── metrics.json
│   ├── predictions_sample.csv
│   └── model_card.md
│
└── docs/
    └── linkedin_post.md
```

---

## 🛠️ Technologies Used

### Data and Analysis

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Regression metrics
- Model artifact management

### Cloud and MLOps Foundation

- AWS SageMaker
- Amazon S3
- SageMaker Hyperparameter Optimization
- Model artifacts
- Model documentation
- API deployment structure

### Deployment

- FastAPI
- Uvicorn
- Docker

---

## 📊 Dataset

The project uses two wine datasets containing physicochemical properties and quality scores.

The datasets are stored in:

```text
data/raw/dataset_1.csv
data/raw/dataset_2.csv
```

The workflow includes:

- loading both datasets;
- standardizing column names;
- merging datasets;
- identifying wine type;
- preparing the target variable;
- splitting the dataset into training, validation and test sets.

---

## 🔍 Exploratory Data Analysis

The exploratory analysis includes:

- distribution analysis;
- correlation map;
- target variable inspection;
- comparison between wine types;
- outlier detection;
- feature relationship analysis.

The goal of this step is to understand the data before modeling and identify potential risks that could affect model performance.

The exploratory notebook is available at:

```text
notebooks/Projeto2.ipynb
```

---

## ⚙️ Data Preprocessing

The preprocessing stage includes:

- feature selection;
- handling categorical information;
- train/validation/test split;
- formatting datasets for XGBoost;
- preparing files for SageMaker training jobs.

The final processed datasets are expected to be stored in:

```text
data/processed/
```

---

## 🤖 Model Training

The main model used in this project is **XGBoost Regressor**, selected due to its strong performance on structured tabular data.

The model is trained to predict the wine quality score based on physicochemical attributes.

---

## 🎛️ Hyperparameter Optimization

A key part of this project is the optimization of XGBoost hyperparameters using **AWS SageMaker Hyperparameter Optimization**.

The optimization process is designed to minimize the validation RMSE.

Configuration summary:

```json
{
  "objective_metric": "validation:rmse",
  "objective_type": "Minimize",
  "max_jobs": 28,
  "max_parallel_jobs": 4
}
```

The best hyperparameters are stored in:

```text
sagemaker/best_hyperparameters.json
```

---

## 📈 Model Evaluation

The final model is evaluated using regression metrics:

| Metric | Purpose |
|---|---|
| MSE | Penalizes larger errors |
| MAE | Average absolute prediction error |
| RMSE | Error measure in the same scale as the target |

Current evaluation artifact:

```text
artifacts/metrics.json
```

Current metrics:

```json
{
  "model": "XGBoost Regressor",
  "target": "wine_quality",
  "problem_type": "regression",
  "metrics": {
    "mse": 0.4564,
    "mae": 0.3867,
    "rmse": 0.6756
  },
  "main_metric": "rmse",
  "evaluation_dataset": "test",
  "pipeline_status": "completed"
}
```

---

## 📦 Artifacts

The project includes the following artifacts:

```text
models/xgboost-model
artifacts/metrics.json
artifacts/model_card.md
sagemaker/best_hyperparameters.json
docs/linkedin_post.md
```

These artifacts provide traceability between training, evaluation, documentation and deployment.

---

## 🧾 Model Card

The model card is available at:

```text
artifacts/model_card.md
```

It documents:

- intended use;
- input features;
- target variable;
- model type;
- evaluation metrics;
- limitations;
- risks;
- responsible use;
- deployment readiness.

This improves transparency and supports responsible machine learning practices.

---

## 🚀 API Deployment

The project includes a FastAPI application for serving predictions.

API file:

```text
app/main.py
```

Expected model artifact path:

```text
models/xgboost-model
```

### Available endpoints

```http
GET /health
POST /predict
```

### Example request

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "wine_type": "red"
}
```

### Expected response

```json
{
  "predicted_quality": 5.873,
  "model": "xgboost",
  "status": "success"
}
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd wine-quality-xgboost-sagemaker-mlops
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate environment

Linux/Mac:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Open the notebook

```bash
jupyter notebook notebooks/Projeto2.ipynb
```

---

## 🚀 Running the API Locally

### Start the API

```bash
uvicorn app.main:app --reload
```

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": "red"
  }'
```

---

## 🐳 Running the API with Docker

### Build image

```bash
docker build -t wine-quality-api .
```

### Run container

```bash
docker run -p 8000:8000 wine-quality-api
```

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": "red"
  }'
```

---

## 🔐 AWS Configuration

This project should not expose AWS credentials, bucket names, account IDs or access keys.

Use environment variables for sensitive configuration:

```bash
SAGEMAKER_BUCKET=<your-bucket-name>
AWS_REGION=<your-region>
```

Do not commit:

```text
.aws/
credentials
access_key
secret_key
account_id
real bucket names
```

---

## 📝 LinkedIn Publication

A concise version of this case study is available at:

```text
docs/linkedin_post.md
```

The LinkedIn post is designed to summarize the project without exposing unnecessary implementation details.

The GitHub repository remains the complete technical reference.

---

## 🧭 Roadmap

### Data Engineering

- [x] Organize raw datasets;
- [x] Add project structure;
- [x] Add notebook;
- [ ] Create reusable preprocessing script;
- [ ] Generate processed datasets.

### Machine Learning

- [x] Train XGBoost model;
- [x] Run SageMaker HPO workflow;
- [x] Store final metrics in `artifacts/metrics.json`;
- [x] Store best hyperparameters in `sagemaker/best_hyperparameters.json`;
- [x] Create model card.

### MLOps and Deployment

- [x] Create FastAPI app;
- [x] Add Dockerfile;
- [x] Add local prediction endpoint;
- [ ] Add automated tests;
- [ ] Add CI/CD roadmap;
- [ ] Add monitoring roadmap;
- [ ] Add cloud deployment option.

---

## ⚠️ Limitations

This project is intended as a portfolio case study and technical demonstration.

Current limitations:

- the model is trained on a public structured dataset;
- no real-time production monitoring is implemented yet;
- deployment is structured as a local/API demonstration;
- AWS resources should be recreated by the user in their own account;
- credentials and cloud identifiers are intentionally excluded;
- the API assumes the model artifact is available at `models/xgboost-model`.

---

## 🧠 Key Takeaways

This project demonstrates that a strong Machine Learning solution is more than a trained model.

A production-oriented workflow requires:

- clear business framing;
- reproducible preprocessing;
- structured experimentation;
- hyperparameter optimization;
- model evaluation;
- artifact management;
- model documentation;
- deployment readiness.

> In Machine Learning Engineering, the model is only one part of the system. The real value comes from building a reproducible and reliable workflow around it.
