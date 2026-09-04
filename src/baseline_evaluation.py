"""
Baseline Evaluation — Breast Cancer Classifier
------------------------------------------------
Establishes a rigorous baseline (in-distribution performance) before any
distribution shift is introduced. This is Stage 2 of the research project:
metrics first, shift experiments come after.

Metrics computed:
  - Accuracy
  - Precision, Recall, F1 (for the MALIGNANT class specifically — this is
    the clinically critical class; missing a malignant case is far costlier
    than a false alarm on a benign one)
  - ROC-AUC
  - 5-fold cross-validated accuracy (single train/test split can be noisy;
    CV gives a more trustworthy estimate + a sense of variance)
  - Full classification report + confusion matrix

Results are saved to baseline_metrics.json so they can be directly compared
against post-shift results later.
"""

import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ----------------------------
# Load data
# ----------------------------
data = load_breast_cancer()
data_X, data_y = data.data, data.target
# NOTE: in this dataset, 0 = malignant, 1 = benign

Xtr, Xte, ytr, yte = train_test_split(
    data_X, data_y,
    test_size=0.2,
    random_state=42,
    stratify=data_y  # preserves class ratio in both splits — important for imbalanced medical data
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(Xtr, ytr)

y_pred = model.predict(Xte)
y_proba = model.predict_proba(Xte)[:, 0]  # probability of class 0 = malignant

# ----------------------------
# Metrics (pos_label=0 because MALIGNANT is the class we care most about catching)
# ----------------------------
accuracy = accuracy_score(yte, y_pred)
precision_malignant = precision_score(yte, y_pred, pos_label=0)
recall_malignant = recall_score(yte, y_pred, pos_label=0)
f1_malignant = f1_score(yte, y_pred, pos_label=0)
# roc_auc_score expects the probability of the "positive" class as sklearn defines it (label 1).
# We flip perspective: use probability of malignant (0) but tell it malignant is our target by inverting.
roc_auc = roc_auc_score(1 - yte, y_proba)  # 1-yte flips labels so malignant=1 for this calculation

fpr, tpr, thresholds = roc_curve(1 - yte, y_proba)

cv_scores = cross_val_score(model, data_X, data_y, cv=5, scoring="accuracy")

cm = confusion_matrix(yte, y_pred)
report = classification_report(yte, y_pred, target_names=["malignant", "benign"])

# ----------------------------
# Report
# ----------------------------
print("=" * 60)
print("BASELINE EVALUATION (in-distribution, no shift)")
print("=" * 60)
print(f"Accuracy:                {accuracy:.4f}")
print(f"Precision (malignant):   {precision_malignant:.4f}")
print(f"Recall (malignant):      {recall_malignant:.4f}   <-- most clinically critical")
print(f"F1-score (malignant):    {f1_malignant:.4f}")
print(f"ROC-AUC:                 {roc_auc:.4f}")
print()
print(f"5-fold CV accuracy:      {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print()
print("Confusion matrix:")
print("                 predicted malignant  predicted benign")
print(f"actual malignant        {cm[0][0]:>4}                {cm[0][1]:>4}")
print(f"actual benign           {cm[1][0]:>4}                {cm[1][1]:>4}")
print()
print(report)

# ----------------------------
# Save for later comparison against shifted-data results
# ----------------------------
baseline_results = {
    "accuracy": accuracy,
    "precision_malignant": precision_malignant,
    "recall_malignant": recall_malignant,
    "f1_malignant": f1_malignant,
    "roc_auc": roc_auc,
    "cv_accuracy_mean": cv_scores.mean(),
    "cv_accuracy_std": cv_scores.std(),
    "confusion_matrix": cm.tolist(),
    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
}

with open("baseline_metrics.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("\nSaved baseline_metrics.json — this is our reference point for shift experiments.")
