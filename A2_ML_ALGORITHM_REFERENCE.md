# ML ALGORITHM REFERENCE — ASSIGNMENT-ALIGNED IMPLEMENTATION RECIPES

Use only the recipe relevant to the scenario. These are patterns, not a requirement to include all algorithms.

## 1. Common Python/EDA
- Typical imports: `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt`.
- Use `df.shape`, `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()`.
- Visualisations: histogram, scatter plot, boxplot, correlation heatmap (if seaborn is installed; otherwise use matplotlib), pairplot/scatter matrix only when useful and available.
- Make each graph readable with title, axis labels and legend as applicable.
- For generated data, set a seed and create a DataFrame with explicit columns.

## 2. Preprocessing
- Numeric missing values: `SimpleImputer(strategy="median")` or a clear pandas median fill. For supervised models, fit imputation on training data only.
- Categorical missing values: most-frequent imputation when categorical features exist.
- Scaling: `StandardScaler` for distance-sensitive algorithms and models where scale affects coefficients; `MinMaxScaler` only where appropriate.
- Train/test split: `train_test_split`; use `stratify=y` for classification when class distribution permits.
- Never fit a scaler on all rows before splitting for supervised evaluation.

## 3. Regression
- Identify a continuous target.
- Baseline: `LinearRegression()` or `Ridge`/`Lasso` where relevant to the question.
- Multiple Linear Regression uses multiple predictor columns; do not confuse feature count with multiple target variables.
- Metrics: `mean_absolute_error`, `mean_squared_error` (take square root for RMSE), `r2_score`.
- Visuals: actual-vs-predicted scatter with a y=x reference line; residual plot; optionally feature/coefficient plot.
- Improvement options: compare baseline with Ridge/Lasso/ElasticNet where covered; feature selection if justified; cross-validation/GridSearchCV only where the relevant assignment covers it. Do not use classification LogisticRegression for continuous marks/consumption.
- For regression regularization, use the correct regression estimator (`Ridge`, `Lasso`, `ElasticNet`), not `LogisticRegression` penalties.

## 4. Classification
- Identify categorical/binary target and class labels.
- Baseline may be `LogisticRegression(max_iter=1000)` or a suitable Decision Tree, depending on scenario.
- Other assignment-aligned models: Perceptron, DecisionTreeClassifier, RandomForestClassifier.
- Metrics: accuracy, precision, recall, F1, confusion matrix; ROC-AUC only for binary/multiclass probability settings where supported.
- Use `classification_report` and confusion matrix; use `ConfusionMatrixDisplay` or matplotlib heatmap if available.
- For disease prediction, recall/sensitivity may matter because false negatives can be costly; do not optimize accuracy alone. Discuss precision/recall trade-off.
- Improvement: L1/L2/ElasticNet regularization for Logistic Regression where compatible; feature importance/selection; Decision Tree depth analysis; Random Forest comparison/tuning with GridSearchCV; class balancing only if class imbalance is observed or explicitly requested.
- For `LogisticRegression` L1 use a compatible solver such as `liblinear` for binary classification; ElasticNet requires `saga` and `l1_ratio`. Verify sklearn version compatibility and use a simple supported configuration.

## 5. Clustering
- No target labels are used to train the clustering model.
- Scale numeric features before distance-based clustering.
- K-Means: `KMeans(n_clusters=k, n_init=10, random_state=42)`; record `inertia_`.
- Choose k using elbow (inertia) and silhouette over valid k values (2 through a modest maximum less than sample count).
- Hierarchical: `AgglomerativeClustering(linkage="ward")`; Ward requires Euclidean distance. Plot dendrogram using scipy only if available and suitable; otherwise use sklearn clustering and a scatter/PCA plot.
- DBSCAN: `DBSCAN(eps=..., min_samples=...)`; report number of non-noise clusters and noise points (`label == -1`). Do not call noise points "clusters".
- Metrics: silhouette (higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher better), only when at least 2 clusters and fewer clusters than samples exist. Handle invalid metric conditions.
- Visuals: elbow/silhouette plots, PCA 2D scatter when useful, cluster profile table (feature means by cluster).
- Improvement: compare K-Means with hierarchical or DBSCAN when shape/density assumptions justify it; justify k/linkage/eps. Do not claim a cluster is "best" from one metric alone.

## 6. Anomaly/outlier detection
- Assignment-aligned statistical/distance techniques include IQR fences, Z-scores, Euclidean distance from centroid after standardisation, Mahalanobis distance using covariance and inverse covariance, and chi-square thresholding; DBSCAN can identify noise points.
- Distinguish univariate outliers from multivariate anomalies. A point can be ordinary in each feature separately but unusual in its joint relationship.
- For Euclidean distance, standardise features before calculating distances. State the threshold strategy (top-k, fixed z/distance threshold, or IQR of distances) and do not choose one silently.
- Mahalanobis distance: `sqrt((x-mu).T @ inv(cov) @ (x-mu))`; use a numerically stable inverse/pseudoinverse if covariance is singular, and explain it. For d dimensions, chi-square threshold uses `chi2.ppf(0.975, df=d)` only when the assumptions are suitable.
- Report threshold, number flagged and row indices. Plot original feature space with flagged points highlighted and a histogram of distances with threshold where useful.
- Compare Euclidean and Mahalanobis flags/rankings. Mahalanobis accounts for covariance/correlation; Euclidean does not.
- DBSCAN can flag noise (`-1`) but is sensitive to scaling and eps/min_samples. Do not describe it as a calibrated probability of attack.
- No labels: do not report supervised accuracy, precision or recall. With known labels, use suitable classification metrics and keep labels out of unsupervised fitting.

## 7. Model optimization and validation
- Assignment 8 patterns: Decision Tree max_depth analysis; Random Forest; OOB score where appropriate; feature importances; GridSearchCV with cv=5.
- Assignment 9 patterns: elbow, silhouette, Davies-Bouldin, Calinski-Harabasz; K-Means vs hierarchical; DBSCAN comparison.
- Assignment 10 patterns: Logistic Regression L1/L2/ElasticNet; coefficient path; Lasso coefficient ranking; reduced feature set; GridSearchCV; constrained design report.
- Assignment 10's LogisticRegression regularization details are classification-specific. Do not copy its LogisticRegression implementation into regression tasks.
- `GridSearchCV` must be fitted on training data only. Use `best_estimator_` and evaluate once on held-out test data.
- Use small parameter grids that are practical on CPU. Avoid huge searches.
