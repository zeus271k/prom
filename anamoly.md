Act as a Senior Python ML Engineer. Write a complete, self-contained, runnable Python script for an UNSUPERVISED ANOMALY DETECTION scenario.

[EXAM QUESTION]:
"Paste the 2-3 sentence question here"

Generate code following this STRICT 4-PART SPECIFICATION to satisfy the 40-mark rubric:

# --- SECTION 1: DATA ACQUISITION & EDA (10 MARKS) ---
- Set np.random.seed(42).
- Generate a pandas DataFrame 'df' of 400 rows representing normal network/system operations (traffic_volume, duration, packet_rate).
- Inject 20 realistic extreme outlier rows (5% anomaly rate) with abnormal spikes in traffic and failed attempts.
- Print df.shape, df.info(), df.describe().
- Plot a 1x2 subplot:
  Subplot 1: Boxplot of numerical features showing presence of extreme outliers.
  Subplot 2: Scatter plot of feature 1 vs feature 2 highlighting extreme spikes.
  plt.tight_layout(); plt.show().

# --- SECTION 2: BASELINE MODEL IMPLEMENTATION (10 MARKS) ---
- Preprocess data using RobustScaler (to prevent outliers from skewing scaling).
- Instantiate and fit baseline IsolationForest(contamination=0.05, random_state=42) on scaled data.
- Predict anomaly flags: -1 for anomaly, 1 for normal.
- Assign flags: df['Anomaly_Baseline'] = iso_forest.predict(X_scaled).
- Print "Baseline Isolation Forest model fitted successfully."

# --- SECTION 3: MODEL EVALUATION (10 MARKS) ---
- Compute anomaly statistics:
  * Count of normal vs anomalous instances: (df['Anomaly_Baseline'] == -1).sum()
  * Detected anomaly percentage vs expected contamination.
  * Anomaly score percentiles using iso_forest.decision_function(X_scaled).
- Use PCA(n_components=2) to project data to 2D.
- Plot 2D scatter plot: normal points in blue, detected anomalies highlighted in red.
  plt.tight_layout(); plt.show().

# --- SECTION 4: TECHNIQUE MODIFICATION (10 MARKS) ---
- Technical upgrade: Implement an alternative density-based algorithm, OneClassSVM(kernel='rbf', nu=0.05).
- Fit OneClassSVM on X_scaled and predict anomalies: df['Anomaly_SVM'] = ocsvm.predict(X_scaled).
- Print a comparison table: Total anomalies flagged by Isolation Forest vs One-Class SVM, and their overlap percentage.
- Print 2 lines of technical justification: why comparing tree-based isolation with kernel hyperplane separation improves detection confidence in cybersecurity.

OUTPUT CONSTRAINTS:
- Output ONLY pure, executable Python code inside a single ```python block.
- Zero markdown explanations outside code comments.
- ZERO placeholder comments ('# TODO', '...'). Include every import explicitly.