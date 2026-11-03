# Qwen2.5-Coder:7B — ML Practical Exam Prompt Toolkit

## Purpose
This package contains three alternative prompt architectures for a 3-hour, 40-mark, scenario-based Machine Learning practical. It is designed for a terminal-based local LLM (Qwen2.5-Coder:7B) running through llama.cpp or Ollama, with 16 GB RAM and no dedicated GPU.

The user's exam rubric:
1. Data acquisition and EDA — 10 marks
2. Supervised or unsupervised technique implementation — 10 marks
3. Model evaluation — 10 marks
4. Modify/improve the technique based on requirements — 10 marks

The ten ML lab assignments are the technical scope. Do not assume an additional syllabus document exists.

## Contents

### Strategy A — Modular architecture (`Strategy_A_Modular/`)
Attach the common master rules, algorithm reference, and execution controller, plus the relevant scenario module. Debug prompt is optional and used only if needed.
- `A1_MASTER_PROMPT.md`
- `A2_ML_ALGORITHM_REFERENCE.md`
- `A3_EXAM_EXECUTION_PROMPT.md`
- `A4_DEBUG_VALIDATION_PROMPT.md`
- `A5_REGRESSION_MODULE.md`
- `A6_CLASSIFICATION_MODULE.md`
- `A7_CLUSTERING_MODULE.md`
- `A8_ANOMALY_MODULE.md`

### Strategy B — Independent prompts (`Strategy_B_Standalone/`)
Each scenario prompt is self-contained. Attach only the matching prompt and paste the scenario.
- `B1_REGRESSION_STANDALONE.md`
- `B2_CLASSIFICATION_STANDALONE.md`
- `B3_CLUSTERING_STANDALONE.md`
- `B4_ANOMALY_STANDALONE.md`

### Strategy C — Hybrid architecture (`Strategy_C_Hybrid/`)
Attach concise global rules + algorithm reference + execution controller + exactly one scenario module. Debug prompt is optional.
- `C1_GLOBAL_RULES.md`
- `C2_COMPACT_ALGORITHM_REFERENCE.md`
- `C3_EXECUTION_CONTROLLER.md`
- `C4_REGRESSION_MODULE.md`
- `C5_CLASSIFICATION_MODULE.md`
- `C6_CLUSTERING_MODULE.md`
- `C7_ANOMALY_MODULE.md`
- `C8_DEBUG_VALIDATION.md`

### Shared resources (`Shared/`)
- `EXAM_SCENARIOS.md` — the five source scenarios.
- `STAGED_EXECUTION_GUIDE.md` — one-shot versus staged generation.
- `DEBUG_AND_REPAIR_PROMPT.md` — reusable repair prompt.
- `VALIDATION_CHECKLIST.md` — rubric and code checks.
- `STRATEGY_COMPARISON_SCORECARD.md` — fair A/B/C testing sheet.
- `TERMINAL_USAGE_EXAMPLES.md` — llama.cpp/Ollama-neutral attachment examples.

## Which prompt should I use?
- Start with Strategy A if you want explicit separation of rules, technical reference and workflow.
- Start with Strategy B if you want one self-contained prompt with fewer attachments.
- Start with Strategy C if you want a balance: concise shared files plus one specialized module.
- Test all three using the same scenario, model settings, context size and seed before deciding which works best.

## Important accuracy notes
- Context windows are measured in tokens, not words. The actual configured context limit must be checked in the lab; do not assume it is 8192.
- The model cannot execute Python in the described terminal workflow. It must not claim that generated code ran or invent metric values. It can perform static reasoning checks only.
- If it suggests a package, verify it is installed in the lab. The user may install permitted packages, but the model must not assume network access.
- The prompt should favor simple, syllabus-aligned implementations over advanced or decorative techniques.
- For generated synthetic data, the model must state that the data is synthetic and choose plausible relationships. Do not fabricate observed results.
- If a question specifies a dataset, use it rather than generating a replacement.
- No prompt can guarantee correct code from a 7B model. Run the code, inspect errors, and use the repair prompt.

## Recommended initial test
Use Q2 (disease classification) for all three strategies, then Q1 or Q5 (regression), Q3 (clustering), and Q4 (anomaly detection). Compare execution errors, missing rubric items, correctness and manual repair effort.

## Assignment scope
The uploaded lab assignments are the source of allowed techniques. The prompt files intentionally avoid treating every method as mandatory for every question. Examples explicitly reflected in the supplied materials include:
- Python, NumPy, pandas, descriptive statistics and plotting.
- EDA, missing values, scaling/normalization and data transformation.
- IQR, Z-score, Euclidean and Mahalanobis outlier detection and chi-square thresholds.
- Linear and multiple linear regression, evaluation and diagnostics.
- Logistic regression, Perceptron, Decision Tree and Random Forest.
- K-Means, hierarchical/agglomerative clustering, DBSCAN, PCA and internal cluster metrics.
- Regularization (Ridge/L2, Lasso/L1, ElasticNet), feature importance/selection and GridSearchCV/cross-validation.

Do not add an algorithm merely because it is popular. If the assignment materials do not establish that a method is in scope, prefer a listed technique or state the uncertainty.
