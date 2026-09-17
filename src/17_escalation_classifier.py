"""
Escalation Classifier
---------------------

Purpose:
    Predict whether a customer message should be:

        AUTO-HANDLE
    or
        HUMAN

The model learns from the human-verified true_escalation labels
in the golden set.

Method:
    1. Load the final golden set.
    2. Split into training and test sets.
    3. Convert customer messages into TF-IDF features.
    4. Train Logistic Regression.
    5. Predict escalation on the held-out test set.
    6. Report precision, recall, F1, and accuracy.
    7. Save predictions and reasons.

IMPORTANT:
    This is an experimental escalation classifier.
    The held-out test set is not used for training.
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

# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FILE PATHS
# ============================================================

# Final human-verified golden set.
GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

# Output file for escalation predictions.
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "escalation_predictions.csv"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

# Make sure the results directory exists.
RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING GOLDEN SET")
print("=" * 70)

# Check that the golden set exists.
if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


# Load the final golden set.
df = pd.read_csv(
    GOLDEN_FILE
)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "customer_message",
    "true_escalation",
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

# Remove rows without the required fields.
df = df.dropna(
    subset=[
        "customer_message",
        "true_escalation",
    ]
).copy()


# Clean customer messages.
df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)


# Convert the escalation label to a real boolean.
#
# The CSV may contain True/False as strings, so we normalize
# them explicitly.
df["true_escalation"] = (
    df["true_escalation"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "true": True,
            "false": False,
        }
    )
)


# Remove any rows where escalation could not be interpreted.
df = df.dropna(
    subset=[
        "true_escalation"
    ]
).copy()


print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# INPUTS AND TARGET
# ============================================================

# Customer support messages.
X = df["customer_message"]

# Human-verified escalation labels.
y = df["true_escalation"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Use the same split strategy as the previous experiments.
#
# Stratification preserves the approximate TRUE/FALSE ratio
# between train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


print("\n")
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training examples: {len(X_train)}"
)

print(
    f"Test examples: {len(X_test)}"
)


# ============================================================
# BUILD ESCALATION MODEL
# ============================================================

# The model consists of:
#
# TF-IDF:
#     Converts the text into numerical features.
#
# Logistic Regression:
#     Predicts whether escalation should be TRUE or FALSE.
#
# class_weight="balanced" helps because the dataset contains
# more FALSE labels than TRUE labels.
escalation_model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
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
# TRAIN MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING ESCALATION CLASSIFIER")
print("=" * 70)

# Train only on the training split.
escalation_model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICT TEST SET
# ============================================================

# Predict TRUE/FALSE escalation labels.
y_pred = escalation_model.predict(
    X_test
)


# Get probabilities so we can see confidence.
y_probability = (
    escalation_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)


print("\n")
print("=" * 70)
print("ESCALATION RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "AUTO-HANDLE",
            "HUMAN",
        ],
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        False,
        True,
    ],
)


print("\nConfusion matrix:")

print(
    pd.DataFrame(
        cm,
        index=[
            "Actual AUTO-HANDLE",
            "Actual HUMAN",
        ],
        columns=[
            "Predicted AUTO-HANDLE",
            "Predicted HUMAN",
        ],
    )
)


# ============================================================
# BUILD PREDICTION OUTPUT
# ============================================================

prediction_df = pd.DataFrame(
    {
        "customer_message": X_test.values,
        "true_escalation": y_test.values,
        "predicted_escalation": y_pred,
        "human_probability": y_probability,
    }
)


# ============================================================
# ADD DECISION TEXT
# ============================================================

# Convert boolean predictions into the human-readable decision
# required by the support-agent design.
prediction_df["decision"] = (
    prediction_df["predicted_escalation"]
    .map(
        {
            True: "HUMAN",
            False: "AUTO-HANDLE",
        }
    )
)


# ============================================================
# ADD REASON
# ============================================================

# Keep the reason conservative.
#
# This is not an LLM-generated explanation. It is an engineering
# explanation based on the model output and should be described
# as such in the report.
prediction_df["reason"] = (
    prediction_df["predicted_escalation"]
    .map(
        {
            True:
                "Escalation classifier predicts that human review is needed.",
            False:
                "Escalation classifier predicts that the request can be handled automatically.",
        }
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

prediction_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("ESCALATION CLASSIFIER COMPLETE")
print("=" * 70)

print(
    "Predictions saved to:"
)

print(
    RESULT_FILE
)