# Supervised Machine Learning Pipeline & AI Safety Evaluation (Python, scikit-learn & pandas)

**Author:** Egor Zvyagin | Melbourne, VIC  
**Contact:** `egor.dobryak1@gmail.com`  
**Profiles:** [LinkedIn](https://linkedin.com/in/egor-zvyagin-843628168) | [GitHub](https://github.com/egor-zvyagin) | [Credly](https://credly.com/users/egor-zvyagin)  
**Specialisation:** Supervised Machine Learning, Feature Engineering, Model Evaluation, Data Science & AI Safety Governance

---

## Executive Summary

This repository presents an end-to-end data science and supervised machine learning pipeline implemented in **Python**, utilizing **scikit-learn**, **pandas**, **numpy**, and **matplotlib**. The project covers the full machine learning lifecycle—from exploratory data analysis (EDA), data cleaning, and feature engineering to model training, hyperparameter optimization, and rigorous statistical evaluation.

In addition to technical model performance, this repository features an enterprise **AI Safety, Ethics, and Governance Evaluation**, establishing concrete frameworks to audit machine learning systems for demographic bias, unfair disparate impact, privacy leakage, and transparency gaps in high-stakes automated decision-making.

---

## Technical Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
|                            DATA PREPARATION & EDA                                 |
+--------------------------+----------------------------+---------------------------+
| • Ingestion & Profiling  | • Missing Value Imputation | • Outlier Detection       |
| • Skewness Analysis      | • Categorical Encoding     | • Feature Scaling         |
+--------------------------+----------------------------+---------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        MODEL TRAINING & OPTIMIZATION                              |
+--------------------------+----------------------------+---------------------------+
| • Baseline Benchmark     | • Multi-Model Exploration  | • Hyperparameter Tuning   |
|   (Logistic Regression)  |   (Random Forests, SVM)    |   (GridSearchCV / K-Fold) |
+--------------------------+----------------------------+---------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     EVALUATION METRICS & DIAGNOSTICS                              |
+-----------------------------------------------------------------------------------+
| • Confusion Matrix (True/False Positives/Negatives)                               |
| • Precision, Recall, F1-Score & ROC-AUC Curves                                    |
| • Bias-Variance Tradeoff & Learning Curves                                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                      AI SAFETY, ETHICS & GOVERNANCE                               |
+-----------------------------------------------------------------------------------+
| • Disparate Impact & Fairness Metrics                                             |
| • Model Explainability (Feature Importance / SHAP Values)                         |
| • Data Privacy (PII Protection & Australian Privacy Principles)                   |
| • Human-in-the-Loop Escalation Protocols                                          |
+-----------------------------------------------------------------------------------+
```

---

## Machine Learning Engineering Highlights

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)
* **Distribution & Variance Inspection:** Analyzed feature variance, target label distributions, and multi-collinearity via Pearson correlation matrices and pairplots.
* **Missing Value Imputation:** Applied median imputation for skewed continuous distributions and mode imputation for categorical attributes, preventing data leakage by fitting transformers strictly on training folds.
* **Categorical Encoding & Feature Scaling:** Implemented `OneHotEncoder` with `handle_unknown='ignore'` for high-cardinality categorical variables, coupled with `StandardScaler` to normalize numeric ranges for distance-based estimators.

### 2. Supervised Model Architecture & Training
* **Benchmarking & Candidate Comparison:** Trained and compared multiple supervised classification algorithms:
  - **Logistic Regression:** Linear decision boundary baseline with L2 Ridge regularization.
  - **Decision Trees:** Non-linear recursive partitioning with pre-pruning constraints (`max_depth`, `min_samples_leaf`).
  - **Random Forest Classifier:** Bagging ensemble aggregating 100+ randomized decision trees to reduce variance and combat overfitting.
  - **Support Vector Classifier (SVC):** Non-linear classification utilizing Radial Basis Function (RBF) kernels with margin tuning (`C` and `gamma`).
* **Stratified K-Fold Cross-Validation:** Utilized 5-fold stratified cross-validation to maintain consistent class proportions across splits, guaranteeing statistical validity.

### 3. Hyperparameter Tuning & Cross-Validation
* Employed `GridSearchCV` to systematically explore hyperparameter combinatorial spaces across trees, estimators, and regularization strengths.
* Monitored training versus validation learning curves to diagnose high bias (underfitting) versus high variance (overfitting).

### 4. Comprehensive Performance Metrics
* **Confusion Matrix Analysis:** Decomposed classification outcomes into True Positives, True Negatives, False Positives (Type I errors), and False Negatives (Type II errors).
* **Precision-Recall Optimization:** Evaluated Macro and Weighted Precision, Recall, and F1-Scores. Tuned decision thresholds to maximize recall in high-stakes detection contexts where False Negatives carry severe risk.
* **ROC-AUC & Precision-Recall Curves:** Generated Receiver Operating Characteristic (ROC) curves, calculating Area Under the Curve (AUC) to assess class separability across variable thresholds.

---

## AI Safety, Ethics & Governance Framework

The accompanying ethical evaluation addresses critical concerns in the operational deployment of algorithmic systems:

### 1. Algorithmic Bias & Fairness Auditing
* **Disparate Impact Analysis:** Audited classification outcomes across sensitive demographic attributes (age, gender, background), applying the Four-Fifths (80%) Rule to identify systemic disadvantage.
* **Fairness Metrics:** Evaluated models against Equalized Odds and Demographic Parity criteria, recommending re-sampling and threshold-adjustment interventions to mitigate historical bias encoded in training corpora.

### 2. Model Interpretability & Explainability
* **Global & Local Interpretability:** Extracted feature importance metrics from tree-based ensembles, validating that predictions align with domain logic rather than spurious correlations or proxy variables.
* **Right to Explanation:** Designed transparent decision receipts that provide human-understandable summaries of key factors influencing individual algorithmic classifications.

### 3. Data Governance, Privacy & Security
* **Data Minimization:** Enforced strict PII stripping (personally identifiable information) during ingestion, ensuring non-essential attributes are expunged prior to model exposure.
* **Regulatory Compliance:** Mapped system data flows to the **Australian Privacy Principles (APPs)** and the Australian Government's **Artificial Intelligence Ethics Framework**.
* **Adversarial Robustness:** Tested model resilience against noisy inputs, boundary perturbations, and data drift, establishing automated retraining triggers upon metric degradation.

### 4. Human-in-the-Loop (HITL) Safeguards
* Defined confidence thresholds requiring ambiguous or borderline predictions (e.g., probability outputs between 45% and 55%) to route automatically to qualified human analysts for manual review.

---

## Python Environment & Setup

Clone the repository and install dependencies in an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/egor-zvyagin/python-ml-stem-pipeline.git
cd python-ml-stem-pipeline

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)
* `scikit-learn>=1.3.0` — Core machine learning models, cross-validation, and metrics
* `pandas>=2.0.0` — High-performance tabular data structures and preprocessing
* `numpy>=1.24.0` — Numerical vector and matrix computation
* `matplotlib>=3.7.0` — Data visualization, confusion matrices, and ROC curves
* `seaborn>=0.12.0` — Statistical distribution plotting and correlation heatmaps
* `scipy>=1.10.0` — Advanced scientific and statistical subroutines
* `joblib>=1.3.0` — Pipeline persistence and parallelized grid search

---

## Included Deliverables

* [`Supervised_Machine_Learning_Pipeline_Report.pdf`](./Supervised_Machine_Learning_Pipeline_Report.pdf): Complete technical report documenting the data preprocessing pipeline, algorithm benchmarking, hyperparameter grid search, and ROC/F1 evaluation curves.
* [`AI_Safety_Governance_and_Ethics_Evaluation.pdf`](./AI_Safety_Governance_and_Ethics_Evaluation.pdf): Detailed ethics and safety governance evaluation, addressing algorithmic fairness, transparency, privacy compliance, and human oversight frameworks.
* [`requirements.txt`](./requirements.txt): Pinned Python package dependencies for exact environment reproducibility.

---

## Professional Reference & Contact

Authored and maintained by **Egor Zvyagin**  
- **Email:** `egor.dobryak1@gmail.com`  
- **LinkedIn:** [linkedin.com/in/egor-zvyagin-843628168](https://linkedin.com/in/egor-zvyagin-843628168)  
- **GitHub:** [github.com/egor-zvyagin](https://github.com/egor-zvyagin)  
- **Credly:** [credly.com/users/egor-zvyagin](https://credly.com/users/egor-zvyagin)
