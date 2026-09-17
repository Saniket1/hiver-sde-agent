"""
Baseline Comparison
-------------------

Purpose:
    Produce a consolidated comparison of:

    1. Majority-class baseline
    2. TF-IDF + Logistic Regression
    3. Sentence Embedding + Nearest Centroid

The goal is to make the model comparison reproducible and
report-ready.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# PROJECT PATH
# ============================================================

# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILES
# ============================================================

# Final human-verified golden set.
GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

# Predictions from the TF-IDF baseline.
TFIDF_FILE = (
    PROJECT_ROOT
    / "results"
    / "baseline_predictions.csv"
)

# Predictions from the embedding classifier.
EMBEDDING_FILE = (
    PROJECT_ROOT
    / "results"
    / "embedding_predictions.csv"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "intent_baseline_comparison.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

# Make sure the golden file exists.
if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


df = pd.read_csv(
    GOLDEN_FILE
)


# ============================================================
# CLEAN DATA
# ============================================================

# Keep only examples with both message and intent label.
df = df.dropna(
    subset=[
        "customer_message",
        "true_intent",
    ]
).copy()


# ============================================================
# RECREATE THE SAME TEST SPLIT
# ============================================================

# Use the exact same split as the earlier experiments.
X = df["customer_message"]

y = df["true_intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# ============================================================
# MAJORITY-CLASS BASELINE
# ============================================================

# Find the most frequent intent in the training data.
majority_intent = (
    y_train.value_counts()
    .idxmax()
)

# Predict that intent for every test example.
majority_predictions = [
    majority_intent
] * len(y_test)


# Calculate majority-baseline metrics.
majority_accuracy = accuracy_score(
    y_test,
    majority_predictions,
)

majority_precision, majority_recall, majority_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        majority_predictions,
        average="macro",
        zero_division=0,
    )
)


# ============================================================
# LOAD TF-IDF PREDICTIONS
# ============================================================

if not TFIDF_FILE.exists():

    raise FileNotFoundError(
        f"TF-IDF prediction file not found:\n{TFIDF_FILE}"
    )


tfidf_df = pd.read_csv(
    TFIDF_FILE
)


# Calculate TF-IDF metrics.
tfidf_accuracy = accuracy_score(
    tfidf_df["true_intent"],
    tfidf_df["predicted_intent"],
)

tfidf_precision, tfidf_recall, tfidf_f1, _ = (
    precision_recall_fscore_support(
        tfidf_df["true_intent"],
        tfidf_df["predicted_intent"],
        average="macro",
        zero_division=0,
    )
)


# ============================================================
# LOAD EMBEDDING PREDICTIONS
# ============================================================

if not EMBEDDING_FILE.exists():

    raise FileNotFoundError(
        f"Embedding prediction file not found:\n{EMBEDDING_FILE}"
    )


embedding_df = pd.read_csv(
    EMBEDDING_FILE
)


# Calculate embedding-model metrics.
embedding_accuracy = accuracy_score(
    embedding_df["true_intent"],
    embedding_df["predicted_intent"],
)

embedding_precision, embedding_recall, embedding_f1, _ = (
    precision_recall_fscore_support(
        embedding_df["true_intent"],
        embedding_df["predicted_intent"],
        average="macro",
        zero_division=0,
    )
)


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "model": "Majority baseline",
            "accuracy": majority_accuracy,
            "macro_precision": majority_precision,
            "macro_recall": majority_recall,
            "macro_f1": majority_f1,
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": tfidf_accuracy,
            "macro_precision": tfidf_precision,
            "macro_recall": tfidf_recall,
            "macro_f1": tfidf_f1,
        },
        {
            "model": "Sentence Embeddings + Nearest Centroid",
            "accuracy": embedding_accuracy,
            "macro_precision": embedding_precision,
            "macro_recall": embedding_recall,
            "macro_f1": embedding_f1,
        },
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("INTENT BASELINE COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)

print("\nMajority intent:")
print(
    majority_intent
)

print("\nSaved to:")
print(
    OUTPUT_FILE
)