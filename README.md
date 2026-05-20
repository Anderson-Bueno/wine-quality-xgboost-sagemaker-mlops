# Wine Quality Prediction with XGBoost and SageMaker HPO

**End-to-end Machine Learning Engineering project for wine quality prediction using XGBoost, AWS SageMaker Hyperparameter Optimization, model evaluation and API deployment.**

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
- API deployment structure;
- documentation for portfolio and technical communication.

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
- train an XGBoost model;
- optimize hyperparameters using SageMaker HPO;
- evaluate the final model using regression metrics;
- organize artifacts and model outputs;
- prepare an API layer for future deployment.

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
API deployment structure
