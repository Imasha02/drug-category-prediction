# drug-category-prediction
📘 Drug Category Prediction Web App

This project is a Machine Learning web application built using XGBoost, CatBoost, Random Forest, and deployed using Streamlit.
The model predicts the drug category (Hallucinogens, Stimulants, Depressants) based on demographic features, personality traits, impulsivity, and sensation-seeking scores.

🚀 Project Overview

The dataset used in this project was:

Already pre-split into train and test sets

Already label-encoded

Contained missing values, which were handled before model training

The goal is to classify drug categories using:

Demographic attributes

Personality trait scores

Impulsivity and sensation-seeking levels

Polynomial & interaction-based engineered features

The final model is a soft-voting ensemble combining three strong classifiers.

🎯 Final Ensemble Composition

The voting classifier uses the following weights:

40% XGBoost

40% CatBoost

20% Random Forest

Soft voting ensures probability-based blending for more robust predictions.

🧠 Machine Learning Pipeline
✔️ 1. Handling Missing Values

Even though the dataset was pre-split and encoded, missing values still needed to be handled.
Appropriate numeric imputation methods were applied to ensure consistent model training.

✔️ 2. Feature Engineering

To improve predictive performance, polynomial and interaction features were created:

Impulsive_SS       = Impulsive × SS
Nscore_Escore      = Nscore × Escore
Impulsive_squared  = Impulsive²
SS_squared         = SS²


These additional features helped capture deeper relationships between traits.

✔️ 3. Models Trained

XGBoostClassifier

CatBoostClassifier

RandomForestClassifier

Each model was trained on the engineered feature set.

✔️ 4. Final Ensemble (VotingClassifier)

A soft-voting classifier was created using the three trained models:

Soft voting allows using predicted probabilities

Weighted averaging improves overall accuracy

XGBoost + CatBoost carry more weight due to stronger performance

🧪 Streamlit Web Application

A full interactive web app was developed using Streamlit.

Features:

User-friendly interface

Inputs for demographic & personality traits

Auto-engineering of the same features used during training

Backend uses saved .pkl model files

Instantly predicts drug category

The following files are included in the project:

ensemble_model.pkl (final model)

label_encoder.pkl

feature_columns.pkl

app.py (Streamlit app)
