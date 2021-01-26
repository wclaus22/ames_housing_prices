# KAGGLE Housing Prices Competition

Notebook with the full solution can be found in the notebooks folder.

The approach is the following:

1. Exploratory data analysis and missing value significance analysis using t-tests.
2. Feature extraction, engineering and imputation using custom `sklearn.pipeline` sub-pipelines
3. Feature selection to mitigate multicollinearity
4. Fit Ridge & Lasso baseline models with `sklearn.model_selection.GridSearchCV`
5. Fit XGBoost model utilizing `xgboost.cv`
6. Model achieves first quartile performance w.r.t. the Kaggle leaderboard without smoothing or manipulating test data predictions
7. Understand predictions and feature importance using SHAP (Shapley Additive Explanations)
