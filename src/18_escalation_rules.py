"""
Improved Rule-Based Escalation Policy
--------------------------------------

Purpose:
    Predict AUTO-HANDLE vs HUMAN using transparent, auditable
    escalation signals derived from the actual golden-set failures.

Design:
    - High-risk signals -> immediate HUMAN
    - Repeated/unresolved support -> HUMAN
    - Repeated delivery failures -> HUMAN
    - Strong dissatisfaction + concrete support problem -> HUMAN
    - Otherwise -> AUTO-HANDLE

This is intentionally transparent rather than a black-box model.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FILE PATHS
# ============================================================

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "rule_escalation_predictions_v2.csv"
)

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HIGH-SEVERITY SIGNALS
# ============================================================

# These signals strongly indicate that a human should review
# the case because they involve fraud, security, theft, or
# explicit management/escalation requests.

HIGH_SEVERITY_PATTERNS = {

    "fraud/security concern": [
        r"\bfraud\b",
        r"\bscam\b",
        r"\bscammed\b",
        r"\bphishing\b",
        r"\bunauthorized\b",
        r"\bunauthorised\b",
        r"\bstolen\b",
        r"\bstole\b",
    ],

    "explicit human escalation request": [
        r"\bmanager\b",
        r"\bsupervisor\b",
        r"\bspeak to someone\b",
        r"\bspeak with someone\b",
        r"\btalk to someone\b",
        r"\btalk to a human\b",
        r"\bhuman agent\b",
        r"\breal person\b",
    ],

    "explicit investigation request": [
        r"\binvestigation\b",
        r"\binvestigate\b",
        r"\bplease investigate\b",
        r"\bformal complaint\b",
        r"\bescalate\b",
        r"\bescalation\b",
    ],
}


# ============================================================
# UNRESOLVED-SUPPORT SIGNALS
# ============================================================

# These signals describe an issue that has already been taken
# to support without successful resolution.

UNRESOLVED_SUPPORT_PATTERNS = [

    r"\bno response\b",
    r"\bno reply\b",
    r"\bno help\b",
    r"\bstill waiting\b",
    r"\bwaiting for a reply\b",
    r"\bwaiting for your reply\b",
    r"\bhasn't helped\b",
    r"\bhasn't help\b",
    r"\bnot been resolved\b",
    r"\bnot resolved\b",
    r"\bunresolved\b",
    r"\bnothing has happened\b",
    r"\bnothing was done\b",
    r"\bnothing can be done\b",
    r"\bcan't help\b",
    r"\bcannot help\b",
    r"\bdidn't help\b",
    r"\bdid not help\b",
    r"\bno one has\b",
    r"\bno one called\b",
    r"\bnever called\b",
    r"\bnever got a reply\b",
    r"\bnever received a reply\b",
    r"\bno response from\b",
]


# ============================================================
# REPEATED-CONTACT SIGNALS
# ============================================================

# These signals indicate that the customer has already tried
# multiple support interactions.

REPEATED_CONTACT_PATTERNS = [

    r"\bcalled 2 times\b",
    r"\bcalled 3 times\b",
    r"\bcalled twice\b",
    r"\bcalled three times\b",
    r"\bthree times\b",
    r"\btwice\b",
    r"\bmultiple times\b",
    r"\bseveral times\b",
    r"\bagain and again\b",
    r"\brepeatedly\b",
    r"\bsecond time\b",
    r"\banother time\b",
    r"\bfor days\b",
    r"\bfor weeks\b",
    r"\bfor a week\b",
    r"\bsince last week\b",
]


# ============================================================
# DELIVERY-FAILURE SIGNALS
# ============================================================

# These patterns capture repeated or serious delivery failures.

DELIVERY_FAILURE_PATTERNS = [

    r"\bpackage.*missing\b",
    r"\bpackages.*missing\b",
    r"\bpackages keep going missing\b",
    r"\bkeep.*missing\b",
    r"\bmarked as delivered.*not received\b",
    r"\bdelivered.*haven't received\b",
    r"\bdelivered.*never received\b",
    r"\bdelivery.*again\b",
    r"\bmissed.*again\b",
    r"\brepeated delivery\b",
    r"\bsecond delivery failure\b",
    r"\bfailed delivery\b",
]


# ============================================================
# STRONG DISSATISFACTION SIGNALS
# ============================================================

# These phrases indicate substantial dissatisfaction.
# We only use them for HUMAN when combined with another concrete
# support/problem signal, to avoid escalating casual complaints.

DISSATISFACTION_PATTERNS = [

    r"\bpathetic service\b",
    r"\bpoor service\b",
    r"\bpoor support\b",
    r"\bwaste of time\b",
    r"\btraining staff\b",
    r"\bunacceptable\b",
    r"\bshocking service\b",
    r"\bterrible service\b",
    r"\bfrustrating\b",
    r"\bvery dissatisfied\b",
    r"\bdisappointed\b",
]


# ============================================================
# CONCRETE-PROBLEM SIGNALS
# ============================================================

# These indicate that the dissatisfaction is attached to an
# actual support problem rather than being purely emotional.

CONCRETE_PROBLEM_PATTERNS = [

    r"\border\b",
    r"\bpackage\b",
    r"\bparcel\b",
    r"\bdelivery\b",
    r"\brefund\b",
    r"\bpayment\b",
    r"\bcharge\b",
    r"\baccount\b",
    r"\bproduct\b",
    r"\breturn\b",
    r"\breplacement\b",
    r"\bsupport\b",
    r"\bcustomer service\b",
    r"\bamazonhelp\b",
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_escalation(message: str):
    """
    Return:

        decision
        reason
    """

    text = str(message).lower()

    # --------------------------------------------------------
    # 1. HIGH-SEVERITY CHECK
    # --------------------------------------------------------

    for category, patterns in HIGH_SEVERITY_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, text):

                return (
                    "HUMAN",
                    f"High-severity signal: {category}."
                )

    # --------------------------------------------------------
    # 2. UNRESOLVED SUPPORT CHECK
    # --------------------------------------------------------

    unresolved_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in UNRESOLVED_SUPPORT_PATTERNS
    )

    if unresolved_match:

        return (
            "HUMAN",
            "Customer reports an unresolved support interaction."
        )

    # --------------------------------------------------------
    # 3. REPEATED CONTACT CHECK
    # --------------------------------------------------------

    repeated_contact_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in REPEATED_CONTACT_PATTERNS
    )

    if repeated_contact_match:

        return (
            "HUMAN",
            "Customer reports repeated contact or repeated failure."
        )

    # --------------------------------------------------------
    # 4. DELIVERY FAILURE CHECK
    # --------------------------------------------------------

    delivery_failure_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in DELIVERY_FAILURE_PATTERNS
    )

    if delivery_failure_match:

        return (
            "HUMAN",
            "Repeated or serious delivery failure detected."
        )

    # --------------------------------------------------------
    # 5. STRONG DISSATISFACTION + CONCRETE PROBLEM
    # --------------------------------------------------------

    dissatisfaction_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in DISSATISFACTION_PATTERNS
    )

    concrete_problem_match = any(
        re.search(
            pattern,
            text
        )
        for pattern in CONCRETE_PROBLEM_PATTERNS
    )

    if (
        dissatisfaction_match
        and concrete_problem_match
    ):

        return (
            "HUMAN",
            "Strong dissatisfaction combined with a concrete support issue."
        )

    # --------------------------------------------------------
    # 6. DEFAULT
    # --------------------------------------------------------

    return (
        "AUTO-HANDLE",
        "No strong escalation signal detected."
    )


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("IMPROVED RULE-BASED ESCALATION EVALUATION")
print("=" * 70)

if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


df = pd.read_csv(
    GOLDEN_FILE
)


# ============================================================
# CLEAN LABELS
# ============================================================

df = df.dropna(
    subset=[
        "customer_message",
        "true_escalation",
    ]
).copy()


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


df = df.dropna(
    subset=[
        "true_escalation"
    ]
).copy()


# ============================================================
# RUN POLICY
# ============================================================

predicted_decisions = []
reasons = []

for message in df["customer_message"]:

    decision, reason = (
        predict_escalation(
            message
        )
    )

    predicted_decisions.append(
        decision
    )

    reasons.append(
        reason
    )


# Convert HUMAN/AUTO-HANDLE into booleans.
predicted_boolean = [
    decision == "HUMAN"
    for decision in predicted_decisions
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    df["true_escalation"],
    predicted_boolean,
)


print("\n")
print("=" * 70)
print("RESULTS")
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
        df["true_escalation"],
        predicted_boolean,
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
    df["true_escalation"],
    predicted_boolean,
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
# SAVE RESULTS
# ============================================================

output_df = pd.DataFrame(
    {
        "customer_message":
            df["customer_message"].values,

        "true_escalation":
            df["true_escalation"].values,

        "predicted_escalation":
            predicted_boolean,

        "decision":
            predicted_decisions,

        "reason":
            reasons,
    }
)


output_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("IMPROVED ESCALATION POLICY COMPLETE")
print("=" * 70)

print(
    "Predictions saved to:"
)

print(
    RESULT_FILE
)