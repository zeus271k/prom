# REGRESSION SCENARIO MODULE

Use this module for continuous numeric targets such as electricity consumption or final examination marks.

## Identify
- Target: one continuous quantity to estimate (e.g. daily_consumption or final_marks).
- Predictors: only information available at prediction time (e.g. household_size, temperature, prior_usage, appliance_usage; or attendance, internal_assessment, assignment_performance, study_hours).
- Do not assume exact CSV column names unless supplied. If synthetic data is needed, define and consistently use clear names.

## Baseline and implementation
- Use LinearRegression for a clear baseline; multiple predictors mean Multiple Linear Regression.
- Split train/test before fitting imputers or scalers. LinearRegression does not require standardisation for ordinary fitting, but scaling may be used where regularization is compared.
- Fit, predict and print actual/predicted sample values.

## Evaluation
- MAE, MSE, RMSE and R².
- Actual-vs-predicted plot with y=x reference.
- Residuals vs predicted plot and residual distribution where useful.
- Explain metric meanings briefly. Do not interpret a negative R² as "negative accuracy".

## Modification/design
Choose an improvement supported by the scenario and assignment scope:
- Compare LinearRegression with Ridge/Lasso/ElasticNet regression if covered by the lab material and relevant.
- Use feature selection only with a justified method and training-only fitting.
- If tuning is used, keep a small grid and use cross-validation on training data.
- Compare baseline and modified models on the same held-out test set with MAE/RMSE/R².
- Do not use LogisticRegression for a continuous target.
- Do not claim regularization improved performance unless computed results show it.
