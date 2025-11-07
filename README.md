# Athlete360 — Predictive Sports Performance and Fatigue Intelligence

An AI-driven sports analytics project designed to predict player performance, fatigue, and injury risk using machine learning and advanced feature engineering.

## 📊 Project Overview

Athlete360 integrates physical workload, fatigue indices, and performance metrics to provide a 360° understanding of athlete readiness.
The project applies predictive modeling and clustering to support data-driven coaching, injury prevention, and individualized training management.

Business Goals:

Predict player performance and fatigue with high accuracy.

Identify players at risk of fatigue or injury before they occur.

Provide actionable insights to coaches and sports scientists for training optimization.

Develop data-driven recommendations for load management and team selection.

##⚙️ Features

Predictive modeling for performance, fatigue, and injury risk

Athlete profiling via K-Means clustering (High-Performance, Balanced, Underperforming groups)

Feature engineering for workload and rolling averages

Fatigue–readiness correlation analysis

Integration of predictive models (LightGBM, RandomForest, Gradient Boosting, Stacking Ensemble)

Visual analytics for model performance and player insights

CRISP-DM workflow applied from data understanding to deployment recommendations

## 🧩 Tech Stack
Category	Tools
Language	Python 3.10+
Libraries	pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn, optuna
Modeling	LightGBM (Tuned), RandomForest, GradientBoosting, Stacking Regressor
EDA & Visualization	matplotlib, seaborn
Clustering	K-Means (Scikit-learn)
Environment	Jupyter Notebook
## 🧪 Workflow Overview

Data Understanding & Preparation

Integration of player performance, fatigue, and injury metrics.

Feature engineering (rolling averages, workload ratios, and normalized performance indicators).

Handling missing values and scaling key features.

## Exploratory Data Analysis (EDA)

Identified strong relationships between workload, fatigue, and performance.

Found inverse correlation between fatigue and readiness.

Clustered players into actionable groups for strategic decisions.

Modeling & Optimization

Baseline models: Ridge, Poisson Regression.

Advanced models: RandomForest, GradientBoosting, LightGBM, Stacking Ensemble.

Hyperparameter tuning with Optuna.

Evaluation metrics: MAE, R².

## Performance Summary

Rank	Model	MAE	R²	Interpretation
1	Final LightGBM (Tuned)	2.819	0.704	Best overall; strong balance between bias and variance.
2	Baseline LightGBM	2.842	0.700	Excellent starting point; improved slightly after tuning.
3	Stacking Ensemble	2.954	0.674	Robust but slightly less accurate than tuned LGBM.
4	Tuned RandomForest	2.954	0.674	Strong tree-based learner; smooth but less generalizable.
5	GradientBoosting	2.981	0.662	Stable; may generalize slightly better on unseen players.
6	Ridge Regression	3.287	0.628	Linear; misses nonlinear fatigue effects.
7	Poisson Regression	3.312	0.612	Underfits complex interactions.

Model Interpretation & Insights

LightGBM tuning improved MAE by ~15% and explained 70% of variance.

Tree-based models captured nonlinear relationships missed by linear baselines.

K-Means clusters revealed distinct athlete profiles for targeted intervention.

## Conclusion (CRISP-DM Phase 6)

Performance depends strongly on workload and efficiency metrics.

Fatigue and injury risk inversely affect readiness.

Tuned LightGBM model provided reliable forecasts for performance and fatigue.

Clustering enabled actionable segmentation of players.

Data integration across metrics offered holistic insights beyond standard box stats.

## Recommendations (CRISP-DM Phase 7)

Adopt predictive analytics in player management.

Use model insights to prevent injuries and optimize rest.

Implement individualized recovery and workload plans.

Foster collaboration between coaches, analysts, and medical staff.

Prioritize model explainability to build trust.

Pilot and scale the Athlete360 system gradually for maximum impact.
