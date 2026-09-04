"""
Breast Cancer Prediction GUI
-----------------------------
Trains a RandomForestClassifier on sklearn's breast cancer dataset, then lets
you view/edit a patient's 30 tumor measurements in a scrollable form and get
a live prediction + confidence score.

Run with:  python breast_cancer_predictor.py
Requires:  pip install scikit-learn
"""

import random
from tkinter import *
from tkinter import ttk, messagebox

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load data & train model
# ----------------------------
data = load_breast_cancer()
data_X, data_y = data.data, data.target
feature_names = data.feature_names  # 30 feature names, in the same order as data_X columns
# NOTE: in this dataset, 0 = malignant, 1 = benign (counter-intuitive, but that's sklearn's encoding)

Xtr, Xte, ytr, yte = train_test_split(
    data_X, data_y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(Xtr, ytr)
accuracy = model.score(Xte, yte)

# ----------------------------
# GUI
# ----------------------------
root = Tk()
root.title("Breast Cancer Prediction AI")
root.geometry("560x700")

Label(
    root,
    text="Breast Cancer Prediction AI",
    font=("Arial", 18, "bold")
).pack(pady=(15, 0))

Label(
    root,
    text=f"Model test accuracy: {accuracy * 100:.2f}%  (RandomForest, held-out test set)",
    font=("Arial", 11)
).pack(pady=(0, 10))

# --- Scrollable frame holding the 30 editable feature fields ---
outer_frame = Frame(root)
outer_frame.pack(fill=BOTH, expand=True, padx=10)

canvas = Canvas(outer_frame, borderwidth=0)
scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
scroll_frame = Frame(canvas)

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

# Enable mouse-wheel scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

# One Entry widget per feature, stored so we can read/write values later
entries = {}
for i, name in enumerate(feature_names):
    row = Frame(scroll_frame)
    row.pack(fill=X, pady=2, padx=5)

    Label(row, text=name.replace("_", " ").title(), width=28, anchor="w").pack(side=LEFT)
    entry = Entry(row, width=15)
    entry.pack(side=LEFT, padx=5)
    entries[name] = entry


def fill_form(sample_values):
    """Fill all entry fields with the given 30 values."""
    for name, val in zip(feature_names, sample_values):
        entries[name].delete(0, END)
        entries[name].insert(0, f"{val:.4f}")


def load_test_patient(index=0):
    """Load a specific patient from the held-out test set (default: first one)."""
    fill_form(Xte[index])
    actual = "Benign" if yte[index] == 1 else "Malignant"
    actual_label.config(text=f"Actual diagnosis (from dataset): {actual}")
    result_label.config(text="Press 'Predict' to run the model.")


def load_random_patient():
    idx = random.randint(0, len(Xte) - 1)
    load_test_patient(idx)


def predict_patient():
    try:
        values = [float(entries[name].get()) for name in feature_names]
    except ValueError:
        messagebox.showerror(
            "Invalid input",
            "All 30 fields must contain numbers. Check for empty or non-numeric fields."
        )
        return

    patient = [values]
    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]  # [P(malignant), P(benign)]

    if prediction == 1:
        result = "🟢 BENIGN"
    else:
        result = "🔴 MALIGNANT"

    result_label.config(
        text=f"{result}\n\n"
             f"Malignant confidence: {probability[0]*100:.2f}%\n"
             f"Benign confidence: {probability[1]*100:.2f}%"
    )


# --- Buttons ---
button_frame = Frame(root)
button_frame.pack(pady=10)

Button(
    button_frame, text="Load First Test Patient", font=("Arial", 11),
    command=lambda: load_test_patient(0), width=20
).grid(row=0, column=0, padx=5)

Button(
    button_frame, text="Load Random Patient", font=("Arial", 11),
    command=load_random_patient, width=20
).grid(row=0, column=1, padx=5)

Button(
    root, text="Predict", font=("Arial", 14, "bold"),
    command=predict_patient, width=20, height=2, bg="#4CAF50", fg="white"
).pack(pady=10)

actual_label = Label(root, text="", font=("Arial", 10, "italic"))
actual_label.pack()

result_label = Label(root, text="Load a patient, then press Predict.", font=("Arial", 13), justify=LEFT)
result_label.pack(pady=10)

# Start with a sample patient already filled in so the form isn't empty
load_test_patient(0)

root.mainloop()
