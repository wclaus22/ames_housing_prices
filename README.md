# KAGGLE Housing Prices Competition

Notebook with the full solution can be found in the notebooks folder.

The approach is the following:

1. Feature sxtraction and engineering using `sklearn.pipeline` and custom pipelines.
2. Feature selection using correlation coefficients w.r.t. the output.
   - Numerical feature selection.
   - Categorical feature selection.
3. Reduce multicollinearity by analysis of correlation coefficients between features.
4. Fit ridge baseline model with `sklearn.model_selection.GridSearchCV`.
5. Fit XGBoost model utilizing `xgboost.cv`.
6. Model achieves first quartile performance w.r.t. the Kaggle leaderboard without smoothing or manipulating test data predictions.
7. Understand predictions and feature importance using SHAP (Shapley Additive Explanations)
