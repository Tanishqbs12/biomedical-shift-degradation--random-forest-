"""
Real-World Distribution Shift Experiment — Golub (1999) Leukemia Dataset
---------------------------------------------------------------------------
Unlike the synthetic-noise experiment on the breast cancer dataset, this uses
GENUINE real-world distribution shift: the original Golub et al. (1999) study
collected the training set (38 bone marrow samples) and test set (34 samples,
including some peripheral blood specimens) from DIFFERENT reference labs using
different sample preparation protocols and, in some cases, different tissue
types (bone marrow vs peripheral blood).

Task: classify Acute Lymphoblastic Leukemia (ALL) vs Acute Myeloid Leukemia
(AML) from 7,129 gene expression measurements.

We train ONCE on golub_train and evaluate the FROZEN model on golub_test --
no retraining, no synthetic perturbation. Any performance gap reflects real
biological/technical distribution shift between labs/protocols.

Because n=38 train / p=7129 genes is extremely high-dimensional and
low-sample, we first apply a standard feature-selection step (select the
genes most correlated with the training labels) -- this mirrors what the
original Golub paper and most microarray ML literature does, and avoids a
meaningless model in a 7129-dimensional space with 38 points.
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42
N_GENES = 50  # top genes selected by ANOVA F-value on the TRAINING set only

# ----------------------------
# Load real data
# ----------------------------
train_df = pd.read_csv("golub_train.csv")
test_df = pd.read_csv("golub_test.csv")

gene_cols = [c for c in train_df.columns if c != "label"]

Xtr_raw = train_df[gene_cols].values
Xte_raw = test_df[gene_cols].values

le = LabelEncoder()
ytr = le.fit_transform(train_df["label"])  # AML=0, ALL=1 (alphabetical)
yte = le.transform(test_df["label"])
aml_label = list(le.classes_).index("AML")
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ----------------------------
# Preprocessing: scale + feature selection FIT ONLY ON TRAIN
# (critical -- fitting on test data would leak information and defeat the
# entire point of testing generalization to a genuinely unseen distribution)
# ----------------------------
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_raw)
Xte_scaled = scaler.transform(Xte_raw)  # transform only, using train statistics

selector = SelectKBest(score_func=f_classif, k=N_GENES)
Xtr_sel = selector.fit_transform(Xtr_scaled, ytr)
Xte_sel = selector.transform(Xte_scaled)

# ----------------------------
# Train FROZEN model on train cohort only
# ----------------------------
model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300)
model.fit(Xtr_sel, ytr)

# In-sample (train cohort) cross-validated performance, for reference
cv_scores = cross_val_score(model, Xtr_sel, ytr, cv=5, scoring="accuracy")

# ----------------------------
# Evaluate on the REAL shifted cohort (golub_test)
# ----------------------------
y_pred = model.predict(Xte_sel)
y_proba = model.predict_proba(Xte_sel)[:, aml_label]

acc = accuracy_score(yte, y_pred)
prec = precision_score(yte, y_pred, pos_label=aml_label, zero_division=0)
rec = recall_score(yte, y_pred, pos_label=aml_label, zero_division=0)
f1 = f1_score(yte, y_pred, pos_label=aml_label, zero_division=0)
auc = roc_auc_score((yte == aml_label).astype(int), y_proba)

# ----------------------------
# Shift detectability: domain classifier on train-cohort vs test-cohort
# features (same 50 selected genes), completely ignoring class labels
# ----------------------------
X_combined = np.vstack([Xtr_sel, Xte_sel])
y_domain = np.concatenate([np.zeros(len(Xtr_sel)), np.ones(len(Xte_sel))])
domain_clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
domain_auc_scores = cross_val_score(domain_clf, X_combined, y_domain, cv=5, scoring="roc_auc")
domain_auc = domain_auc_scores.mean()

# ----------------------------
# Report
# ----------------------------
print("=" * 65)
print("REAL-WORLD SHIFT: trained on golub_train, tested on golub_test")
print("(different labs / protocols / some different tissue types)")
print("=" * 65)
print(f"Train-cohort 5-fold CV accuracy:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Test-cohort accuracy (AML class): {acc:.4f}")
print(f"Test-cohort precision (AML):      {prec:.4f}")
print(f"Test-cohort recall (AML):         {rec:.4f}")
print(f"Test-cohort F1 (AML):             {f1:.4f}")
print(f"Test-cohort ROC-AUC:              {auc:.4f}")
print(f"Domain-classifier AUC (train vs test cohort, unsupervised): {domain_auc:.4f}")
print()
print(f"Performance gap (CV acc - test acc): {cv_scores.mean() - acc:.4f}")

results = {
    "dataset": "Golub 1999 (real train/test cohort split)",
    "n_genes_selected": N_GENES,
    "train_cv_accuracy_mean": cv_scores.mean(),
    "train_cv_accuracy_std": cv_scores.std(),
    "test_accuracy": acc,
    "test_precision_aml": prec,
    "test_recall_aml": rec,
    "test_f1_aml": f1,
    "test_roc_auc": auc,
    "domain_classifier_auc": domain_auc,
}
with open("real_shift_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved real_shift_results.json")
