"""
Mitigation & Adaptation Strategies Analysis
--------------------------------------------

Compares:

1. Base Random Forest
2. Platt Scaling (post-hoc calibration)
3. Robust Training through data augmentation

The experiment evaluates whether these approaches can improve
model reliability under progressively stronger synthetic
distribution shifts.

Important design choices:
- The base and robust models use the SAME training subset.
- Both models use the SAME Random Forest configuration.
- The calibration set is kept separate from the training data.
- The final test set is never used for training or calibration.
- Feature standard deviations used for synthetic noise are
  calculated ONLY from the training subset.
- Robust training uses a FIXED augmentation severity of 0.75.
- Evaluation is performed across multiple shift severities.
"""

import json
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA & CREATE DATA SPLITS
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target


# First split:
# 80% development/training data
# 20% final untouched test data

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# Second split:
# Split the training data into:
#
# 75% actual model-training data
# 25% calibration data
#
# The calibration data is NOT used to train the classifier.

X_tr_sub, X_calib, y_tr_sub, y_calib = train_test_split(
    Xtr,
    ytr,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=ytr
)


# ============================================================
# 2. CALCULATE FEATURE SCALE FROM TRAINING DATA ONLY
# ============================================================

# IMPORTANT:
# We do NOT calculate this from the entire dataset.
# Otherwise information from the final test cohort would
# influence the synthetic shift generation.

feature_stds = X_tr_sub.std(axis=0)


# ============================================================
# 3. SYNTHETIC DISTRIBUTION SHIFT
# ============================================================

def apply_shift(X_input, severity, seed=RANDOM_STATE):
    """
    Apply a synthetic distribution shift.

    Noise magnitude is proportional to the standard deviation
    of each feature in the TRAINING subset.

    severity = 0.0
        No artificial shift.

    severity = 0.5
        Moderate synthetic shift.

    severity = 1.0
        Noise standard deviation equals the training
        standard deviation of each feature.

    Higher values represent progressively stronger
    synthetic distribution shifts.
    """

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=feature_stds * severity,
        size=X_input.shape
    )

    return X_input + noise


# ============================================================
# 4. EXPECTED CALIBRATION ERROR
# ============================================================

def compute_ece(y_true, y_prob, n_bins=5):
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between:
        - predicted probability
        - observed frequency

    Lower ECE = better calibration.
    """

    bin_boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (
                (y_prob >= bin_lower) &
                (y_prob <= bin_upper)
            )
        else:
            in_bin = (
                (y_prob > bin_lower) &
                (y_prob <= bin_upper)
            )

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:

            acc_in_bin = np.mean(
                y_true[in_bin]
            )

            conf_in_bin = np.mean(
                y_prob[in_bin]
            )

            ece += (
                np.abs(
                    acc_in_bin - conf_in_bin
                )
                * prop_in_bin
            )

    return float(ece)


# ============================================================
# 5. TRAIN BASE MODEL
# ============================================================

# IMPORTANT:
# The base and robust models use exactly the same
# Random Forest configuration.

model_base = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200
)

model_base.fit(
    X_tr_sub,
    y_tr_sub
)


# ============================================================
# 6. PLATT SCALING
# ============================================================

# First obtain predictions on the separate calibration set.

p_calib = model_base.predict_proba(
    X_calib
)[:, 1]


# Platt scaling learns a logistic transformation of
# the original model probabilities.

platt_scaler = LogisticRegression(
    random_state=RANDOM_STATE
)

platt_scaler.fit(
    p_calib.reshape(-1, 1),
    y_calib
)


# ============================================================
# 7. ROBUST TRAINING THROUGH DATA AUGMENTATION
# ============================================================

def generate_augmented_data(
    X_input,
    y_input,
    severity=0.75,
    multiplier=4
):
    """
    Create augmented training data by adding synthetic
    feature-scaled Gaussian perturbations.

    IMPORTANT:
    The function receives ONLY X_tr_sub and y_tr_sub.

    Therefore the robust model has access to the same
    original training observations as the base model.

    severity=0.75 is deliberately fixed.

    This experiment therefore asks:

        Does exposure to a moderate synthetic shift
        during training improve robustness across
        a range of stronger test-time shifts?
    """

    X_aug_list = [X_input]
    y_aug_list = [y_input]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for _ in range(multiplier):

        noise = rng.normal(
            loc=0.0,
            scale=feature_stds * severity,
            size=X_input.shape
        )

        X_noisy = X_input + noise

        X_aug_list.append(
            X_noisy
        )

        y_aug_list.append(
            y_input
        )

    return (
        np.vstack(X_aug_list),
        np.hstack(y_aug_list)
    )


# IMPORTANT:
# Use EXACTLY the same training subset as the base model.

X_aug, y_aug = generate_augmented_data(
    X_tr_sub,
    y_tr_sub,
    severity=0.75,
    multiplier=4
)


# Same Random Forest configuration as base model.

robust_model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200
)

robust_model.fit(
    X_aug,
    y_aug
)


# ============================================================
# 8. EVALUATE ACROSS SHIFT SEVERITIES
# ============================================================

severities = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0
]


results = []


print("\n" + "=" * 100)

print(
    "             MITIGATION & ROBUSTNESS EXPERIMENTAL RESULTS"
)

print("=" * 100)

print(
    f"{'Severity':<10} | "
    f"{'Base Acc':<10} | "
    f"{'Base ECE':<10} | "
    f"{'Platt ECE':<10} | "
    f"{'Robust Acc':<11} | "
    f"{'Robust ECE':<11}"
)

print("-" * 100)


for sev in severities:

    # --------------------------------------------------------
    # Create ONE shifted test set.
    #
    # Both models receive exactly the same shifted data.
    # --------------------------------------------------------

    X_shifted = apply_shift(
        Xte,
        sev,
        seed=RANDOM_STATE
    )


    # ========================================================
    # BASE MODEL
    # ========================================================

    p_base = model_base.predict_proba(
        X_shifted
    )[:, 1]

    acc_base = model_base.score(
        X_shifted,
        yte
    )

    ece_base = compute_ece(
        yte,
        p_base
    )


    # ========================================================
    # PLATT-CALIBRATED MODEL
    # ========================================================

    p_platt = platt_scaler.predict_proba(
        p_base.reshape(-1, 1)
    )[:, 1]

    ece_platt = compute_ece(
        yte,
        p_platt
    )


    # ========================================================
    # ROBUST MODEL
    # ========================================================

    p_robust = robust_model.predict_proba(
        X_shifted
    )[:, 1]

    acc_robust = robust_model.score(
        X_shifted,
        yte
    )

    ece_robust = compute_ece(
        yte,
        p_robust
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"{sev:<10.1f} | "
        f"{acc_base * 100:<9.1f}% | "
        f"{ece_base:<10.4f} | "
        f"{ece_platt:<10.4f} | "
        f"{acc_robust * 100:<10.1f}% | "
        f"{ece_robust:<11.4f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.append({

        "severity": sev,

        "base_accuracy": acc_base,

        "base_ece": ece_base,

        "platt_ece": ece_platt,

        "robust_accuracy": acc_robust,

        "robust_ece": ece_robust

    })


# ============================================================
# 9. SAVE EXPERIMENT RESULTS
# ============================================================

print("=" * 100)
print()


with open(
    "mitigation_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    "Saved mitigation_results.json successfully."
)

print()
print(
    "Experimental design:"
)

print(
    "- Base model trained on X_tr_sub."
)

print(
    "- Robust model trained on the SAME X_tr_sub."
)

print(
    "- Robust augmentation severity fixed at 0.75."
)

print(
    "- Platt scaling fitted only on X_calib."
)

print(
    "- Xte remains completely untouched until final evaluation."
)

print(
    "- Feature noise scale calculated only from training data."
)
