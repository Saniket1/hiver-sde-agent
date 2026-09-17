"""
Finalize the Golden Evaluation Set
-----------------------------------

Purpose:
    Create the final golden set using only examples that were
    actually human-verified.

The assignment requires 150-250 hand-labelled examples.
We currently have 191 verified examples, so the 9 ambiguous/
incomplete messages can be excluded rather than guessed.
"""

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

# Find the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# INPUT / OUTPUT FILES
# ------------------------------------------------------------

# Current file containing all 200 candidates and human labels.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.csv"
)

# Final file containing only human-verified examples.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

# Read the current golden-set file.
df = pd.read_csv(INPUT_FILE)


# ------------------------------------------------------------
# KEEP ONLY HUMAN-VERIFIED EXAMPLES
# ------------------------------------------------------------

# A row is considered human-verified when true_intent contains
# one of the valid taxonomy labels.
valid_intents = {
    "Order_Delivery_Tracking",
    "Delivery_Delay_or_Missed_Date",
    "Order_Cancellation_or_Change",
    "Missing_Wrong_or_Damaged_Item",
    "Returns_Refunds_or_Replacements",
    "Payment_or_Charge_Issue",
    "Account_Login_or_Security",
    "Product_Availability_or_Pricing",
    "Product_or_Service_Information",
    "Prime_Video_Content_Availability",
}

# Keep only rows with a valid final human intent.
final_df = df[
    df["true_intent"]
    .astype(str)
    .isin(valid_intents)
].copy()


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

# Count the final number of verified examples.
final_count = len(final_df)

# Count examples that still lack human verification.
excluded_count = len(df) - final_count

print("=" * 70)
print("FINAL GOLDEN SET VALIDATION")
print("=" * 70)

print(f"Original candidates: {len(df)}")
print(f"Human-verified examples: {final_count}")
print(f"Excluded/unverified examples: {excluded_count}")


# ------------------------------------------------------------
# CHECK REQUIRED RANGE
# ------------------------------------------------------------

# The assignment requires 150-250 hand-labelled examples.
if 150 <= final_count <= 250:

    print(
        "\nSTATUS: PASS"
    )

    print(
        "The final golden set is within the required "
        "150-250 example range."
    )

else:

    print(
        "\nSTATUS: WARNING"
    )

    print(
        "The final golden set is outside the required range."
    )


# ------------------------------------------------------------
# CHECK FOR MISSING LABELS
# ------------------------------------------------------------

# Verify that the final set has no missing true_intent values.
missing_intents = (
    final_df["true_intent"]
    .isna()
    .sum()
)

# Verify that the final set has no missing escalation values.
missing_escalation = (
    final_df["true_escalation"]
    .isna()
    .sum()
)

print(
    f"\nMissing true_intent labels: "
    f"{missing_intents}"
)

print(
    f"Missing true_escalation labels: "
    f"{missing_escalation}"
)


# ------------------------------------------------------------
# SAVE FINAL GOLDEN SET
# ------------------------------------------------------------

# Save only the verified examples.
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nFinal golden set saved to:"
)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# INTENT DISTRIBUTION
# ------------------------------------------------------------

print("\nIntent distribution:")

print(
    final_df["true_intent"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)

if (
    final_count >= 150
    and missing_intents == 0
    and missing_escalation == 0
):

    print(
        "GOLDEN SET READY"
    )

else:

    print(
        "GOLDEN SET NEEDS REVIEW"
    )

print("=" * 70)