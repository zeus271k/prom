# ANOMALY DETECTION SCENARIO MODULE

Use this module when the goal is to flag unusual observations that deviate from normal behavior (e.g. network traffic anomalies).

## Identify and prepare
- Candidate numeric features may include traffic volume, connection duration and failed login attempts.
- Do not assume exact units or column names unless supplied.
- Do not label anomalies as confirmed cyberattacks; call them flagged unusual observations.
- If labels are not provided, treat the task as unsupervised/statistical anomaly detection.

## Baseline and implementation
- Start with assignment-aligned IQR or standardized Euclidean distance from the centroid.
- Explain whether the method is univariate or multivariate.
- Standardize features before distance calculations.
- A multivariate Mahalanobis method is an appropriate comparison when feature covariance matters.
- DBSCAN can be included as a density-based comparison when suitable.

## Evaluation
- Print the chosen threshold, number flagged and row indices/identifiers.
- Plot feature-space scatter with flagged points highlighted and distance histogram with threshold.
- Compare Euclidean and Mahalanobis rankings/flags if both are used.
- If known ground-truth labels are supplied, compare with them using suitable metrics. Without labels, do not report accuracy, precision, recall or F1.
- Use chi-square threshold only with stated assumptions and degrees of freedom equal to feature dimension.

## Modification/design
- Compare a univariate IQR baseline with multivariate Mahalanobis or DBSCAN where justified.
- Alternatively compare threshold strategies (top-k, IQR of distances, justified chi-square threshold).
- Explain covariance awareness: Mahalanobis accounts for correlations and differing scales after appropriate setup; Euclidean does not.
- Report trade-offs: false alarms, missed anomalies, sensitivity to threshold and scaling.
- Do not claim flagged network events are attacks without investigation or labels.
