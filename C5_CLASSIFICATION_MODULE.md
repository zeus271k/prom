# CLASSIFICATION SCENARIO MODULE

Use this module for categorical targets such as disease status (positive/negative).

## Identify
- Target: confirmed diagnostic category / class.
- Predictors: patient features such as age, blood pressure, glucose, cholesterol, only if supplied or explicitly generated.
- For binary disease prediction, define class labels clearly (0 = no disease, 1 = disease) when generating data.
- Never treat patient data or synthetic results as clinical evidence.

## Baseline and implementation
- LogisticRegression is a suitable baseline for binary classification.
- Use stratified train/test split where possible.
- Fit preprocessing on training data only.
- Print class distribution, predictions and predicted probabilities where needed.

## Evaluation
- Accuracy, precision, recall, F1, confusion matrix and ROC-AUC when valid.
- Disease context: discuss recall/sensitivity and false negatives; do not optimize accuracy alone.
- Plot confusion matrix and ROC curve where appropriate.
- If one class is absent in a split or AUC is undefined, handle it explicitly.

## Modification/design
Use assignment-aligned changes:
- Compare DecisionTreeClassifier and/or RandomForestClassifier with baseline where justified.
- Analyze tree depth / overfitting where a Decision Tree is used.
- Consider class_weight="balanced" only when class imbalance is observed or the scenario justifies it.
- Use LogisticRegression L1/L2/ElasticNet regularization with compatible solvers; for L1 use a compatible solver such as liblinear in binary classification, and for ElasticNet use saga with l1_ratio.
- Use Lasso-style LogisticRegression coefficients for feature selection only after scaling and using training data. Do not confuse LogisticRegression L1 with regression Lasso.
- GridSearchCV on training data only; evaluate best estimator on held-out test data.
- Compare baseline and modified models using recall, F1, ROC-AUC and precision trade-off. Do not promise an improvement.
