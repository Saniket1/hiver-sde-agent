"""
Human Agreement / Annotation Consistency Check
------------------------------------------------

Purpose:
    Measure consistency of the human annotation process.

Method:
    1. Load the 191 final golden examples.
    2. Randomly select 20 examples.
    3. Hide their original labels.
    4. Ask the human annotator to label them again.
    5. Compare second-pass labels with original labels.
    6. Calculate agreement percentage.

Important:
    This is an intra-annotator consistency check because the same
    annotator performs both passes.

It is NOT the same as independent multi-annotator agreement.
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

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

AGREEMENT_FILE = (
    PROJECT_ROOT
    / "results"
    / "human_agreement_check.csv"
)


# ============================================================
# FROZEN INTENTS
# ============================================================

INTENTS = [
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
]


# ============================================================
# LOAD GOLDEN SET
# ============================================================

# Check that the final golden set exists.
if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


# Load the final human-verified dataset.
df = pd.read_csv(
    GOLDEN_FILE
)


# ============================================================
# RANDOM SAMPLE
# ============================================================

# Use a fixed seed so the same 20 examples are used if the
# script needs to be resumed.
sample = (
    df.sample(
        n=20,
        random_state=2026,
    )
    .reset_index(drop=True)
)


# ============================================================
# RESULTS CONTAINER
# ============================================================

results = []


# ============================================================
# DISPLAY INSTRUCTIONS
# ============================================================

print("=" * 70)
print("HUMAN AGREEMENT / CONSISTENCY CHECK")
print("=" * 70)

print(
    "\nYou will re-label 20 examples."
)

print(
    "The original labels are hidden."
)

print(
    "Do NOT look at the golden_set_final.csv while doing this."
)

print(
    "\nIntent options:"
)

for number, intent in enumerate(
    INTENTS,
    start=1,
):

    print(
        f"{number:2d}. {intent}"
    )


# ============================================================
# LABEL EACH SAMPLE
# ============================================================

for position, row in sample.iterrows():

    print("\n")
    print("=" * 70)

    print(
        f"Example {position + 1}/20"
    )

    print("=" * 70)

    print("\nCUSTOMER MESSAGE:")

    print(
        row["customer_message"]
    )

    # --------------------------------------------------------
    # GET SECOND-PASS INTENT
    # --------------------------------------------------------

    while True:

        choice = input(
            "\nEnter intent number 1-10: "
        ).strip()

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= 10:

                second_label = (
                    INTENTS[number - 1]
                )

                break

        print(
            "Invalid choice. "
            "Enter a number from 1 to 10."
        )

    # --------------------------------------------------------
    # GET SECOND-PASS ESCALATION
    # --------------------------------------------------------

    while True:

        escalation_choice = input(
            "Escalation? T = TRUE / F = FALSE: "
        ).strip().lower()

        if escalation_choice in [
            "t",
            "f",
        ]:

            second_escalation = (
                escalation_choice == "t"
            )

            break

        print(
            "Please enter T or F."
        )

    # --------------------------------------------------------
    # COMPARE WITH ORIGINAL LABEL
    # --------------------------------------------------------

    original_intent = (
        row["true_intent"]
    )

    original_escalation = (
        str(
            row["true_escalation"]
        )
        .strip()
        .lower()
        == "true"
    )

    intent_agreement = (
        second_label
        == original_intent
    )

    escalation_agreement = (
        second_escalation
        == original_escalation
    )

    overall_agreement = (
        intent_agreement
        and escalation_agreement
    )

    # --------------------------------------------------------
    # SAVE RESULT IN MEMORY
    # --------------------------------------------------------

    results.append(
        {
            "golden_candidate_id":
                row["golden_candidate_id"],

            "customer_message":
                row["customer_message"],

            "original_intent":
                original_intent,

            "second_pass_intent":
                second_label,

            "intent_agreement":
                intent_agreement,

            "original_escalation":
                original_escalation,

            "second_pass_escalation":
                second_escalation,

            "escalation_agreement":
                escalation_agreement,

            "overall_agreement":
                overall_agreement,
        }
    )


# ============================================================
# SAVE AGREEMENT RESULTS
# ============================================================

agreement_df = pd.DataFrame(
    results
)

agreement_df.to_csv(
    AGREEMENT_FILE,
    index=False,
)


# ============================================================
# CALCULATE AGREEMENT
# ============================================================

intent_agreement_rate = (
    agreement_df[
        "intent_agreement"
    ]
    .mean()
)

escalation_agreement_rate = (
    agreement_df[
        "escalation_agreement"
    ]
    .mean()
)

overall_agreement_rate = (
    agreement_df[
        "overall_agreement"
    ]
    .mean()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("AGREEMENT RESULTS")
print("=" * 70)

print(
    f"Sample size: {len(agreement_df)}"
)

print(
    f"Intent agreement: "
    f"{intent_agreement_rate:.2%}"
)

print(
    f"Escalation agreement: "
    f"{escalation_agreement_rate:.2%}"
)

print(
    f"Overall agreement: "
    f"{overall_agreement_rate:.2%}"
)

print("\nSaved to:")

print(
    AGREEMENT_FILE
)

print("\nIMPORTANT:")
print(
    "This is intra-annotator consistency, not independent "
    "multi-human annotator agreement."
)
