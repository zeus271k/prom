Act as a Senior Python ML Engineer. Write a complete, self-contained, runnable Python script for an UNSUPERVISED CLUSTERING scenario.

[EXAM QUESTION]:
"Paste the 2-3 sentence question here"

Generate code following this STRICT 4-PART SPECIFICATION to satisfy the 40-mark rubric:

# --- SECTION 1: DATA ACQUISITION & EDA (10 MARKS) ---
- Set np.random.seed(42).
- Generate an unlabelled pandas DataFrame 'df' of 300 rows with 4 purchasing/behavioral features (e.g., purchase_freq, annual_spend, discount_pct, variety_score).
- Introduce 5 np.nan values and impute using SimpleImputer(strategy='mean').
- Print df.shape, df.info(), df.describe(), and df.isnull().sum().
- Plot a 1x2 subplot:
  Subplot 1: sns.histplot(df.iloc[:, 0], kde=True) with title "Feature 1 Distribution".
  Subplot 2: sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f') with title "Correlation Heatmap".
  plt.tight_layout(); plt.show().

# --- SECTION 2: BASELINE MODEL IMPLEMENTATION (10 MARKS) ---
- Scale all features using StandardScaler: X_scaled = scaler.fit_transform(df).
- Fit baseline KMeans(n_clusters=3, random_state=42, n_init=10) on X_scaled.
- Assign labels: df['Cluster_Baseline'] = kmeans.labels_.
- Print "Baseline K-Means (K=3) fitted successfully."

# --- SECTION 3: MODEL EVALUATION (10 MARKS) ---
- Calculate and print:
  * Silhouette Score: silhouette_score(X_scaled, kmeans.labels_)
  * Davies-Bouldin Index: davies_bouldin_score(X_scaled, kmeans.labels_)
  * Cluster sizes: df['Cluster_Baseline'].value_counts()
- Use PCA(n_components=2) to reduce X_scaled to 2 dimensions for visualization.
- Plot 2D PCA scatter plot with points colored by 'Cluster_Baseline' and plot cluster centroids.
  plt.tight_layout(); plt.show().

# --- SECTION 4: TECHNIQUE MODIFICATION (10 MARKS) ---
- Requirement: The company needs the data-driven optimal number of clusters rather than guessing K=3.
- Compute Inertia and Silhouette scores for K in range(2, 8).
- Plot 1x2 subplot: Elbow Curve (Inertia vs K) and Silhouette Curve (Silhouette Score vs K).
- Automatically pick optimal K (highest silhouette score) and re-fit KMeans(n_clusters=optimal_k, random_state=42).
- Print a comparison table: Baseline K=3 Silhouette Score vs Optimal K Silhouette Score.
- Print 2 lines of domain justification: how optimal K provides mathematically validated customer segments.

OUTPUT CONSTRAINTS:
- Output ONLY pure, executable Python code inside a single ```python block.
- Zero markdown explanations outside code comments.
- ZERO placeholder comments ('# TODO', '...'). Include every import explicitly.