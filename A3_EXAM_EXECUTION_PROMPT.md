# EXAM EXECUTION CONTROLLER

Read the full attached master rules and relevant algorithm/scenario reference before generating code.

## Step 1 — Parse the question
Extract:
- real-world objective;
- available dataset or requirement to generate data;
- target variable, if supervised;
- candidate input features;
- task type: regression, classification, clustering or anomaly detection;
- explicit constraints and requested outputs.

Do not infer a target label for an unsupervised task. If the question is ambiguous, state the most reasonable interpretation and proceed without asking unless a missing detail makes code impossible.

## Step 2 — Select a method
Choose the simplest assignment-aligned baseline. Give a one-line reason. Identify one justified improvement for Section 4. Do not use all algorithms merely to show breadth.

## Step 3 — Decide output mode
Default to ONE complete script if it can fit comfortably in the response and context. If likely to be truncated, use staged generation. Never stop halfway through a code block. In staged mode, clearly state the variables produced and the exact next-stage request.

## Step 4 — Implement four sections
### SECTION 1 (10): Data acquisition and EDA
Load or generate data; inspect; report missing values; preprocess; visualize; define X/y where applicable.
### SECTION 2 (10): Technique implementation
Train baseline or fit unsupervised method; predict/assign; print useful outputs.
### SECTION 3 (10): Evaluation
Calculate correct task-specific metrics and produce evaluation plots. Do not fabricate results.
### SECTION 4 (10): Modification/design
Implement a justified improvement; compare it against baseline using consistent data and valid metrics; state whether the requirement is met.

## Step 5 — Static self-check before answering
Check:
- imports and API names;
- variable definitions and ordering;
- feature/target separation;
- split-before-fit preprocessing;
- compatible model and metric;
- no test leakage;
- valid parameter values;
- valid plotting calls;
- all four sections present;
- no invented result values;
- manageable CPU runtime.

Do not claim execution. End with concise assumptions and a run checklist.
