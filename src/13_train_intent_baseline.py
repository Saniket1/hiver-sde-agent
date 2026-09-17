"""
TF-IDF + Logistic Regression Intent Baseline
---------------------------------------------

Purpose:
    Train a simple text-classification baseline using the
    human-verified golden set.

Why this baseline:
    - Simple and reproducible.
    - Fast to train.
    - Provides a meaningful benchmark before using embeddings/LLMs.
    - Gives us a reference point for later model comparisons.

Important:
    The golden set should be used for evaluation, not training.
    Therefore, we split the 191 human-verified examples into
    train and test portions using stratification.
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATH
# ============================================================

# Find the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILE
# ============================================================

# Final human-verified golden set.
GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)


# ============================================================
# RESULT FILE
# ============================================================

# Save the baseline predictions so we can inspect them later.
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "baseline_predictions.csv"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

# Create the results directory if it does not already exist.
RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

# Check that the input file exists.
if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


# Load the final human-verified examples.
df = pd.read_csv(
    GOLDEN_FILE
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "customer_message",
    "true_intent",
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

# Remove rows with missing text or missing labels.
df = df.dropna(
    subset=[
        "customer_message",
        "true_intent",
    ]
).copy()


# Convert text to string and remove unnecessary whitespace.
df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)


# ============================================================
# INPUTS AND TARGET
# ============================================================

# X contains the customer messages.
X = df["customer_message"]

# y contains the human-verified intent labels.
y = df["true_intent"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Use stratification so that every intent represented in the
# split is distributed as evenly as possible.
#
# random_state makes the experiment reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# ============================================================
# BUILD BASELINE PIPELINE
# ============================================================

# The pipeline performs:
#
# 1. TF-IDF:
#       Converts text into numerical features.
#
# 2. Logistic Regression:
#       Learns the relationship between the text features
#       and the intent classes.
#
# This is deliberately simple because it is our baseline.
baseline_model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAINING TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

print(
    f"Total examples: {len(df)}"
)

print(
    f"Training examples: {len(X_train)}"
)

print(
    f"Test examples: {len(X_test)}"
)


# Train the baseline classifier.
baseline_model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICT
# ============================================================

# Generate predictions for the held-out test set.
y_pred = baseline_model.predict(
    X_test
)


# ============================================================
# EVALUATE
# ============================================================

# Overall accuracy.
accuracy = accuracy_score(
    y_test,
    y_pred,
)


print("\n")
print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Get the label order from the trained classifier.
labels = baseline_model[
    "classifier"
].classes_


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)


print("\nConfusion matrix:")

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

# Create a dataframe containing the held-out examples,
# their true labels, and the baseline predictions.
prediction_df = pd.DataFrame(
    {
        "customer_message": X_test.values,
        "true_intent": y_test.values,
        "predicted_intent": y_pred,
    }
)


# Save the prediction file.
prediction_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# FINISH
# ============================================================

print("\nPredictions saved to:")

print(
    RESULT_FILE
)

print("\nBaseline training complete.")