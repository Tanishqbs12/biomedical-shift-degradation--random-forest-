"""
Distribution Shift Experiment — Breast Cancer Classifier
------------------------------------------------------------
Stage 3 of the research project.

We take the FROZEN baseline model (never retrained) and evaluate it on
increasingly shifted versions of the test set. Shift is a SYNTHETIC /
SIMULATED distribution shift: per-feature Gaussian noise, scaled to each
feature's own standard deviation, so all 30 features drift proportionally
to their natural scale. Increasing severity levels represent progressively
stronger synthetic perturbations -- this is a controlled experimental
variable, not a claim about any specific real-world cause (e.g. a particular
hospital, scanner, or population). Real-world shift is investigated
separately using genuine cohort/batch splits on real biotech data.

At each severity level we measure:
  - Accuracy, Precision/Recall/F1 (malignant), ROC-AUC   [performance]
  - Domain-classifier AUC                                 [shift detectability
    -- can we tell original vs shifted data apart WITHOUT using any labels?]

This lets us test the actual research question: does performance degrade
under shift, and can that degradation be anticipated using an unsupervised
signal (domain-classifier AUC) alone?

Results saved to shift_experiment_results.json for the paper's tables/plots.
"""

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ----------------------------
# Recreate the EXACT baseline setup (frozen model, never touches shifted data)
# ----------------------------
data = load_breast_cancer()
data_X, data_y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(
    data_X, data_y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=data_y
)

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(Xtr, ytr)

# Load baseline numbers for direct comparison
with open("baseline_metrics.json") as f:
    baseline = json.load(f)

feature_stds = data_X.std(axis=0)  # per-feature natural scale, used to scale injected noise

# ----------------------------
# Shift simulation
# ----------------------------
def apply_shift(X, severity, seed):
    """
    Add Gaussian noise to each feature, proportional to that feature's std.
    severity=0.0 -> no shift (identical to baseline).
    severity=1.0 -> noise std equal to the feature's own natural std (substantial drift).
    """
    local_rng = np.random.default_rng(seed)
    noise = local_rng.normal(loc=0.0, scale=feature_stds * severity, size=X.shape)
    return X + noise


def domain_classifier_auc(X_source, X_target, seed):
    """
    Train a simple classifier to distinguish "source" (original training
    distribution) from "target" (shifted test data), with NO label information
    used -- purely based on feature distributions.
    AUC ~0.5  -> distributions look identical, shift is not detectable
    AUC ~1.0  -> distributions are trivially separable, shift is severe
    This is a standard unsupervised shift-detection technique.
    """
    X_combined = np.vstack([X_source, X_target])
    y_domain = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y_domain, cv=5, scoring="roc_auc")
    return scores.mean()


# ----------------------------
# Run experiment across severity levels
# ----------------------------
severities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
results = []

for sev in severities:
    Xte_shifted = apply_shift(Xte, sev, seed=RANDOM_STATE)

    y_pred = model.predict(Xte_shifted)
    y_proba = model.predict_proba(Xte_shifted)[:, 0]  # P(malignant)

    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, pos_label=0, zero_division=0)
    rec = recall_score(yte, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(yte, y_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(1 - yte, y_proba)
    except ValueError:
        auc = float("nan")

    # Compare the ORIGINAL (unshifted) test distribution against the SHIFTED
    # test distribution -- this isolates the shift itself, rather than mixing
    # in the ordinary train/test sampling difference.
    shift_auc = domain_classifier_auc(Xte, Xte_shifted, seed=RANDOM_STATE)

    results.append({
        "severity": sev,
        "accuracy": acc,
        "precision_malignant": prec,
        "recall_malignant": rec,
        "f1_malignant": f1,
        "roc_auc": auc,
        "domain_classifier_auc": shift_auc,
    })

    print(f"severity={sev:>4.2f} | acc={acc:.3f} | recall(malig)={rec:.3f} | "
          f"f1={f1:.3f} | roc_auc={auc:.3f} | shift_detect_auc={shift_auc:.3f}")

# ----------------------------
# Save results
# ----------------------------
output = {
    "baseline": baseline,
    "shift_results": results,
}
with open("shift_experiment_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved shift_experiment_results.json")
