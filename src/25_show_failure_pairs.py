"""
Show Intent Failure Pairs
-------------------------

Purpose:
    Read the saved intent failure-analysis file and display:
    1. The most common confusion pairs.
    2. Representative customer examples for each pair.

This is used to identify the top 5 failure modes for the report.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Find the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILE
# ============================================================

# Failure-analysis output created by script 24.
INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "intent_failure_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

# Make sure the failure file exists.
if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Failure-analysis file not found:\n{INPUT_FILE}"
    )


# Load the misclassified examples.
df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# FIND COMMON CONFUSION PAIRS
# ============================================================

print("=" * 70)
print("TOP CONFUSION PAIRS")
print("=" * 70)

# Group errors by:
#     true intent -> predicted intent
#
# Then count how many times each confusion occurred.
confusion_pairs = (
    df
    .groupby(
        [
            "true_intent",
            "predicted_intent",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)


# Display the top 10 confusion pairs.
print(
    confusion_pairs
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# SHOW REPRESENTATIVE EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("REPRESENTATIVE FAILURE EXAMPLES")
print("=" * 70)


# Take the five most frequent confusion pairs.
top_pairs = (
    confusion_pairs
    .head(5)
)


# Loop through each confusion pair.
for _, pair in top_pairs.iterrows():

    true_intent = pair[
        "true_intent"
    ]

    predicted_intent = pair[
        "predicted_intent"
    ]

    count = pair[
        "count"
    ]

    print("\n")
    print("-" * 70)

    print(
        f"TRUE INTENT: {true_intent}"
    )

    print(
        f"PREDICTED INTENT: {predicted_intent}"
    )

    print(
        f"NUMBER OF ERRORS: {count}"
    )

    print("-" * 70)

    # Get up to three real customer examples for this
    # particular confusion pair.
    examples = df[
        (
            df["true_intent"]
            == true_intent
        )
        &
        (
            df["predicted_intent"]
            == predicted_intent
        )
    ].head(3)

    # Display each example.
    for example_number, (_, example) in enumerate(
        examples.iterrows(),
        start=1,
    ):

        print(
            f"\nExample {example_number}:"
        )

        print(
            example["customer_message"]
        )

        # Similarity score is available from the embedding
        # classifier output and is useful for diagnosing
        # borderline predictions.
        if "similarity_score" in example:

            print(
                f"Similarity: "
                f"{example['similarity_score']:.4f}"
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FAILURE PAIR ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Total misclassified examples: {len(df)}"
)