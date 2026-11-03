# CLUSTERING SCENARIO MODULE

Use this module when the goal is to group similar observations and no predefined target labels are supplied (e.g. customer segmentation).

## Identify and prepare
- Use behavior variables such as purchase frequency, spending habits and product preferences if supplied.
- Exclude customer identifiers from distance calculations.
- Encode categorical preferences only with a simple defensible method; do not pretend arbitrary integer codes are meaningful distances.
- Scale numeric features before distance-based clustering.
- Do not create or use a target label for training.

## Baseline and implementation
- K-Means is a suitable baseline for compact, roughly spherical customer groups.
- Use KMeans(n_clusters=k, n_init=10, random_state=42).
- Choose candidate k using elbow/inertia and silhouette over valid k values.
- Also fit AgglomerativeClustering (Ward where suitable) or DBSCAN when comparison is useful and covered by the assignments.
- For DBSCAN, report noise separately.

## Evaluation
- Silhouette score (higher is better), Davies-Bouldin (lower is better), Calinski-Harabasz (higher is better).
- Compute only when the number of clusters is valid (at least 2 and less than number of samples).
- Plot elbow and silhouette; visualize clusters using PCA 2D only as a visualization projection, not as hidden ground truth.
- Print cluster sizes and feature profiles/means.
- Do not use accuracy/F1 without true external labels.

## Modification/design
- Improve selection of k, compare K-Means with hierarchical clustering or DBSCAN, or adjust DBSCAN eps/min_samples based on a small transparent search.
- Use the same scaled features for fair comparison.
- Discuss assumptions: K-Means needs k and favors spherical clusters; hierarchical gives a dendrogram; DBSCAN can identify density-based noise but is sensitive to eps and scaling.
- Select a recommendation based on multiple metrics and interpretability, not one score alone.
