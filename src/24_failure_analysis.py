"""
Intent Failure Analysis
-----------------------

Purpose:
    Analyze the actual mistakes made by the embedding intent
    classifier on the held-out evaluation set.

Outputs:
    1. Detailed misclassification file.
    2. Most common intent confusion pairs.
    3. Representative examples for each confusion pair.

This supports the assignment's required top-5 failure analysis.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FILES
# ============================================================

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "embedding_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "intent_failure_analysis.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("INTENT FAILURE ANALYSIS")
print("=" * 70)

# Check that the embedding prediction file exists.
if not PREDICTIONS_FILE.exists():

    raise FileNotFoundError(
        f"Embedding prediction file not found:\n"
        f"{PREDICTIONS_FILE}"
    )


# Load the held-out predictions.
df = pd.read_csv(
    PREDICTIONS_FILE
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = {
    "customer_message",
    "true_intent",
    "predicted_intent",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        f"Missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# IDENTIFY FAILURES
# ============================================================

# Keep only examples where the predicted intent is different
# from the human-verified intent.
failures = df[
    df["true_intent"]
    !=
    df["predicted_intent"]
].copy()


# ============================================================
# ADD ERROR TYPE
# ============================================================

# A simple human-readable description of the error.
failures["error_type"] = (
    failures["true_intent"]
    + " -> "
    + failures["predicted_intent"]
)


# ============================================================
# SAVE DETAILED FAILURES
# ============================================================

failures.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# COMMON CONFUSION PAIRS
# ============================================================

print("\n")
print("=" * 70)
print("COMMON CONFUSION PAIRS")
print("=" * 70)

confusion_pairs = (
    failures[
        [
            "true_intent",
            "predicted_intent",
        ]
    ]
    .value_counts()
    .reset_index(
        name="count"
    )
)


print(
    confusion_pairs.to_string(
        index=False
    )
)


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("REPRESENTATIVE FAILURE EXAMPLES")
print("=" * 70)


# Show up to three examples for the most frequent confusion pairs.
for _, pair in confusion_pairs.head(10).iterrows():

    true_intent = pair[
        "true_intent"
    ]

    predicted_intent = pair[
        "predicted_intent"
    ]

    print("\n")
    print("-" * 70)

    print(
        f"TRUE: {true_intent}"
    )

    print(
        f"PREDICTED: {predicted_intent}"
    )

    examples = failures[
        (
            failures["true_intent"]
            == true_intent
        )
        &
        (
            failures["predicted_intent"]
            == predicted_intent
        )
    ].head(3)

    for _, example in examples.iterrows():

        print(
            "\nCustomer:"
        )

        print(
            example["customer_message"]
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Total test examples: {len(df)}"
)

print(
    f"Misclassified examples: {len(failures)}"
)

print(
    f"Error rate: "
    f"{len(failures) / len(df):.2%}"
)

print("\nSaved to:")
print(
    OUTPUT_FILE
)