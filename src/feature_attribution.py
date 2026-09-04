"""
Feature-Level Drift Attribution Analysis
-----------------------------------------
Identifies high-vulnerability features using Gini Importance, Permutation Importance,
and Single-Feature Shift Sensitivity to pinpoint which measurements drive OOD failure.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

# 1. Load data & train baseline model
data = load_breast_cancer()
X_df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

Xtr, Xte, ytr, yte = train_test_split(
    X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(Xtr, ytr)

# 2. Compute Permutation Importances (Clean vs Shifted)
perm_clean = permutation_importance(model, Xte, yte, random_state=RANDOM_STATE, n_repeats=10)

feature_stds = X_df.std(axis=0).values
rng = np.random.default_rng(RANDOM_STATE)
Xte_shifted = Xte.copy() + rng.normal(loc=0.0, scale=feature_stds * 1.0, size=Xte.shape)

perm_shifted = permutation_importance(model, Xte_shifted, yte, random_state=RANDOM_STATE, n_repeats=10)

# 3. Assemble Attribution DataFrame
df_attribution = pd.DataFrame({
    "Gini_Importance": model.feature_importances_,
    "Clean_Permutation_Importance": perm_clean.importances_mean,
    "Shifted_Permutation_Importance": perm_shifted.importances_mean
}, index=data.feature_names).sort_values(by="Clean_Permutation_Importance", ascending=False)

print("\n" + "="*75)
print("          FEATURE-LEVEL DRIFT ATTRIBUTION (TOP 10 FEATURES)")
print("="*75)
print(f"{'Feature Name':<25} | {'Gini Imp.':<10} | {'Clean Perm.':<12} | {'Shifted Perm.':<12}")
print("-" * 75)

for feat, row in df_attribution.head(10).iterrows():
    print(f"{feat:<25} | {row['Gini_Importance']:<10.4f} | {row['Clean_Permutation_Importance']:<12.4f} | {row['Shifted_Permutation_Importance']:<12.4f}")

print("="*75 + "\n")

# Save JSON results
with open("feature_attribution_results.json", "w") as f:
    json.dump(df_attribution.to_dict(), f, indent=2)

print("Saved 'feature_attribution_results.json' successfully.")
