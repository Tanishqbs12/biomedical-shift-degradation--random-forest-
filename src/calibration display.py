"""
Uncertainty Quantification & Calibration Analysis
--------------------------------------------------
Computes Expected Calibration Error (ECE) and displays reliability diagrams
to assess whether confidence scores remain reliable under distribution shift.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42

# 1. Load data & train baseline model
data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    data.data, data.target, test_size=0.2, random_state=RANDOM_STATE, stratify=data.target
)

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(Xtr, ytr)
feature_stds = data.data.std(axis=0)

def apply_shift(X, severity, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    return X + rng.normal(loc=0.0, scale=feature_stds * severity, size=X.shape)

def compute_ece(y_true, y_prob, n_bins=5):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)

severities = [0.0, 0.5, 1.0, 1.5, 2.0]
results = {}

print("\n" + "="*65)
print("     UNCERTAINTY & CALIBRATION ANALYSIS (SHIFT EXPERIMENT)")
print("="*65)
print(f"{'Severity':<10} | {'Accuracy':<12} | {'ECE (Calibration Error)':<25}")
print("-" * 65)

plt.figure(figsize=(9, 6))

for sev in severities:
    X_shifted = apply_shift(Xte, sev)
    prob_benign = model.predict_proba(X_shifted)[:, 1]
    
    acc = model.score(X_shifted, yte)
    ece = compute_ece(yte, prob_benign, n_bins=5)
    results[f"severity_{sev}"] = {"accuracy": acc, "ece": ece}
    
    print(f"{sev:<10.1f} | {acc*100:<11.2f}% | {ece:<25.4f}")
    
    prob_true, prob_pred = calibration_curve(yte, prob_benign, n_bins=5)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Severity {sev:.1f} (ECE: {ece:.3f})')

print("="*65 + "\n")

# Plot styling
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability (Benign)', fontsize=12)
plt.ylabel('Fraction of True Positives (Benign)', fontsize=12)
plt.title('Reliability Diagram Across Shift Severities', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Save image and output files
plt.savefig('reliability_diagram.png', dpi=300, bbox_inches='tight')
with open("calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved 'reliability_diagram.png' and 'calibration_results.json'.")
print("Displaying graph window now...")

# Open image pop-up window
plt.show()