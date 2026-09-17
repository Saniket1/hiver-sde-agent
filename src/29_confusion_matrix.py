"""
Hiver SDE Intern Assignment
Intent Confusion Matrix

Purpose:
1. Load the existing 48-example intent evaluation results.
2. Build a confusion matrix from true vs predicted intents.
3. Save the matrix as a PNG image.
4. Print a simple summary to the terminal.

The model does not need to be retrained.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Find the project root automatically.
# This file lives inside the src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Existing evaluation file containing all holdout predictions.
EVALUATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_detailed.csv"
)


# Output image for the confusion matrix.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "confusion_matrix.png"
)


# ============================================================
# 2. LOAD EVALUATION DATA
# ============================================================

# Read the existing evaluation results.
df = pd.read_csv(
    EVALUATION_FILE
)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

# These are the two columns required to build a confusion matrix.
required_columns = [
    "true_intent",
    "predicted_intent",
]


# Check that both columns exist.
missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


# Stop with a clear message if the file schema is unexpected.
if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 4. REMOVE INVALID ROWS
# ============================================================

# Keep only rows where both the true and predicted intents exist.
evaluation_df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# 5. PRINT BASIC INFORMATION
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    f"Evaluation examples: {len(evaluation_df)}"
)

print(
    f"True intents: "
    f"{evaluation_df['true_intent'].nunique()}"
)

print(
    f"Predicted intents: "
    f"{evaluation_df['predicted_intent'].nunique()}"
)


# ============================================================
# 6. CREATE CONFUSION MATRIX
# ============================================================

# Use a fixed label order based on all intents appearing in either
# the true or predicted column.
labels = sorted(
    set(evaluation_df["true_intent"])
    | set(evaluation_df["predicted_intent"])
)


# Create the matplotlib figure.
fig, ax = plt.subplots(
    figsize=(14, 11)
)


# Generate the confusion matrix directly from the true and
# predicted intent columns.
ConfusionMatrixDisplay.from_predictions(
    evaluation_df["true_intent"],
    evaluation_df["predicted_intent"],
    labels=labels,
    display_labels=labels,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
    colorbar=True,
)


# ============================================================
# 7. FORMAT THE FIGURE
# ============================================================

# Add a descriptive title.
ax.set_title(
    "AmazonHelp Intent Classification — Confusion Matrix"
)


# Label the axes.
ax.set_xlabel(
    "Predicted Intent"
)

ax.set_ylabel(
    "True Intent"
)


# Make the layout fit long intent names.
plt.tight_layout()


# ============================================================
# 8. SAVE IMAGE
# ============================================================

# Save the confusion matrix as a PNG.
plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)


# Close the figure after saving.
plt.close(fig)


# ============================================================
# 9. PRINT OUTPUT LOCATION
# ============================================================

print(
    f"\nConfusion matrix saved to:\n{OUTPUT_FILE}"
)

print(
    "\nP1 #17 — Confusion Matrix: COMPLETE"
)