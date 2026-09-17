"""
Fast Human Verification for the Golden Set
-------------------------------------------

Purpose:
    Quickly human-verify the 200 AmazonHelp golden-set examples.

Workflow:
    1. Load golden_ai_labeled.csv.
    2. Use a valid Gemini suggestion when available.
    3. Otherwise use keyword-based suggestion.
    4. If neither produces a suggestion, require manual selection.
    5. ENTER -> accept the displayed suggestion.
    6. 1-10 -> select an intent manually.
    7. E -> toggle escalation TRUE/FALSE.
    8. S -> skip the example.
    9. Save progress after every example.

IMPORTANT:
    AI/rule suggestions are only pre-labels.
    true_intent and true_escalation are the human-verified labels.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Locate the project root automatically based on this script's
# location. The script is inside the "src" folder, so parents[1]
# points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FILE PATHS
# ============================================================

# Existing AI-assisted labeling file.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_ai_labeled.csv"
)

# Final human-verified golden-set file.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.csv"
)


# ============================================================
# FROZEN INTENT TAXONOMY
# ============================================================

# These are the 10 intents already established for the project.
# Do not change these names while labeling the golden set.

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
# INTENT DESCRIPTIONS
# ============================================================

INTENT_DESCRIPTIONS = {
    "Order_Delivery_Tracking":
        "Tracking, location, or status of an order/package.",

    "Delivery_Delay_or_Missed_Date":
        "Late, delayed, or missed promised delivery date.",

    "Order_Cancellation_or_Change":
        "Cancel/change an order or unexpected cancellation.",

    "Missing_Wrong_or_Damaged_Item":
        "Missing, wrong, incomplete, or damaged item/package.",

    "Returns_Refunds_or_Replacements":
        "Return, refund, or replacement request.",

    "Payment_or_Charge_Issue":
        "Payment, charge, billing, or cashback issue.",

    "Account_Login_or_Security":
        "Login, account access, verification, or security issue.",

    "Product_Availability_or_Pricing":
        "Product price, discount, promotion, stock, availability.",

    "Product_or_Service_Information":
        "General product/service information or compatibility.",

    "Prime_Video_Content_Availability":
        "Prime Video movie/show/season/episode availability.",
}


# ============================================================
# KEYWORD RULES
# ============================================================

# These rules provide a quick suggestion when Gemini did not
# successfully produce a label.

KEYWORD_RULES = {

    "Prime_Video_Content_Availability": [
        "prime video",
        "primevideo",
        "movie",
        "movies",
        "show",
        "shows",
        "season",
        "episode",
        "episodes",
        "stream",
        "streaming",
        "watch",
    ],

    "Order_Cancellation_or_Change": [
        "cancel my order",
        "cancel order",
        "cancel",
        "canceled order",
        "cancelled order",
        "change my order",
        "modify my order",
    ],

    "Returns_Refunds_or_Replacements": [
        "refund",
        "refunded",
        "return",
        "returning",
        "replacement",
        "replace",
        "money back",
    ],

    "Missing_Wrong_or_Damaged_Item": [
        "missing item",
        "missing items",
        "missing",
        "wrong item",
        "wrong product",
        "damaged",
        "broken",
        "defective",
        "incomplete",
        "not received",
        "package was empty",
    ],

    "Payment_or_Charge_Issue": [
        "charged",
        "charge",
        "payment",
        "billing",
        "cashback",
        "credit card",
        "debit card",
        "unauthorized charge",
    ],

    "Account_Login_or_Security": [
        "login",
        "log in",
        "logged in",
        "password",
        "account",
        "verification",
        "verify",
        "security",
        "hack",
        "hacked",
        "unauthorized access",
    ],

    "Delivery_Delay_or_Missed_Date": [
        "late",
        "delayed",
        "delay",
        "missed",
        "overdue",
        "past the date",
        "supposed to arrive",
        "should have arrived",
        "not arrived yet",
        "still waiting",
    ],

    "Order_Delivery_Tracking": [
        "where is my order",
        "where is my package",
        "where's my order",
        "where's my package",
        "tracking",
        "track my order",
        "track package",
        "delivery status",
        "package location",
        "order status",
    ],

    "Product_Availability_or_Pricing": [
        "price",
        "pricing",
        "discount",
        "offer",
        "sale",
        "promotion",
        "available",
        "availability",
        "in stock",
        "out of stock",
        "stock",
        "cheaper",
    ],

    "Product_or_Service_Information": [
        "compatible",
        "compatibility",
        "how does",
        "how do i",
        "information",
        "feature",
        "features",
        "what is",
        "what does",
        "can this",
    ],
}


# ============================================================
# RULE-BASED SUGGESTION
# ============================================================

def rule_based_suggestion(message):
    """
    Generate an intent suggestion based on keyword matches.

    Returns:
        best_intent:
            The intent with the most keyword matches, or None
            if no keyword matched.

        best_score:
            Number of matching keywords.
    """

    # Convert the customer message to lowercase so keyword
    # matching is case-insensitive.
    text = str(message).lower()

    # Start every intent with zero keyword matches.
    scores = {
        intent: 0
        for intent in INTENTS
    }

    # Count keyword matches for every intent.
    for intent, keywords in KEYWORD_RULES.items():

        for keyword in keywords:

            if keyword in text:
                scores[intent] += 1

    # Find the intent with the highest number of matches.
    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    # If nothing matched, do not invent an intent.
    if best_score == 0:
        return None, 0

    return best_intent, best_score


# ============================================================
# HUMAN INTENT SELECTION
# ============================================================

def select_intent_manually():
    """
    Show all 10 intents and force the user to select one.

    This function is used when no reliable automatic suggestion
    is available.
    """

    print("\nNO AUTOMATIC SUGGESTION AVAILABLE.")
    print("Please select the correct intent manually:\n")

    # Display all 10 options.
    for number, intent in enumerate(
        INTENTS,
        start=1
    ):

        print(
            f"{number:2d}. {intent}"
        )

        print(
            f"    {INTENT_DESCRIPTIONS[intent]}"
        )

    # Keep asking until a valid number is provided.
    while True:

        choice = input(
            "\nEnter intent number 1-10: "
        ).strip()

        if choice.isdigit():

            number = int(choice)

            if 1 <= number <= 10:

                return INTENTS[number - 1]

        print(
            "Invalid choice. "
            "Please enter a number from 1 to 10."
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("FAST HUMAN GOLDEN-SET VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD AI-LABELED FILE
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df)} golden candidates."
    )

    # --------------------------------------------------------
    # CREATE HUMAN-LABEL COLUMNS
    # --------------------------------------------------------

    # Pandas can infer empty columns as float64.
    # We explicitly make these columns "object" so that text
    # labels and boolean values can safely be stored in them.

    for column in [
        "true_intent",
        "true_escalation",
        "human_review_note",
    ]:

        if column not in df.columns:

            df[column] = pd.Series(
                [pd.NA] * len(df),
                dtype="object"
            )

        else:

            df[column] = df[column].astype("object")

    # --------------------------------------------------------
    # LOAD PREVIOUS HUMAN PROGRESS
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        try:

            existing = pd.read_csv(
                OUTPUT_FILE
            )

            # Make sure the human-label columns can store strings.
            for column in [
                "true_intent",
                "true_escalation",
                "human_review_note",
            ]:

                if column in existing.columns:

                    existing[column] = (
                        existing[column].astype("object")
                    )

            # Use candidate ID to match previous answers.
            if "golden_candidate_id" in existing.columns:

                existing_lookup = existing.set_index(
                    "golden_candidate_id"
                )

                # Copy previous labels into the current dataframe.
                for row_number in range(len(df)):

                    candidate_id = df.at[
                        row_number,
                        "golden_candidate_id"
                    ]

                    if candidate_id in existing_lookup.index:

                        previous = (
                            existing_lookup.loc[
                                candidate_id
                            ]
                        )

                        for column in [
                            "true_intent",
                            "true_escalation",
                            "human_review_note",
                        ]:

                            if column in previous.index:

                                previous_value = (
                                    previous[column]
                                )

                                if pd.notna(previous_value):

                                    df.at[
                                        row_number,
                                        column
                                    ] = previous_value

                print(
                    "Previous human progress loaded."
                )

        except Exception as exc:

            print(
                "Could not load previous final file."
            )

            print(
                f"Reason: {exc}"
            )

    # --------------------------------------------------------
    # REVIEW ALL EXAMPLES
    # --------------------------------------------------------

    total = len(df)

    for row_number in range(total):

        # ----------------------------------------------------
        # SKIP ALREADY VERIFIED EXAMPLES
        # ----------------------------------------------------

        current_label = df.at[
            row_number,
            "true_intent"
        ]

        if (
            pd.notna(current_label)
            and str(current_label).strip() in INTENTS
        ):

            continue

        # ----------------------------------------------------
        # GET CUSTOMER MESSAGE
        # ----------------------------------------------------

        message = str(
            df.at[
                row_number,
                "customer_message"
            ]
        ).strip()

        # ----------------------------------------------------
        # CHECK GEMINI SUGGESTION
        # ----------------------------------------------------

        ai_intent = df.at[
            row_number,
            "ai_suggested_intent"
        ]

        if (
            pd.notna(ai_intent)
            and str(ai_intent).strip() in INTENTS
        ):

            suggested_intent = str(
                ai_intent
            ).strip()

            suggestion_source = "Gemini"

        else:

            # ------------------------------------------------
            # TRY KEYWORD SUGGESTION
            # ------------------------------------------------

            suggested_intent, match_count = (
                rule_based_suggestion(message)
            )

            if suggested_intent is not None:

                suggestion_source = (
                    f"Keyword rules "
                    f"({match_count} match(es))"
                )

            else:

                # ------------------------------------------------
                # NO SUGGESTION
                # ------------------------------------------------
                #
                # IMPORTANT:
                # Do NOT assign a default intent.
                # The human must choose one.
                # ------------------------------------------------

                suggested_intent = None

                suggestion_source = (
                    "No automatic suggestion"
                )

        # ----------------------------------------------------
        # GET ESCALATION SUGGESTION
        # ----------------------------------------------------

        ai_escalation = df.at[
            row_number,
            "ai_suggested_escalation"
        ]

        if pd.notna(ai_escalation):

            escalation = (
                str(ai_escalation)
                .strip()
                .lower()
                == "true"
            )

        else:

            # When no AI escalation exists, start with FALSE.
            escalation = False

        # ----------------------------------------------------
        # DISPLAY CURRENT EXAMPLE
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)

        print(
            f"Example {row_number + 1}/{total}"
        )

        print("=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print(message)

        print("\nSUGGESTED INTENT:")

        if suggested_intent is not None:

            print(
                suggested_intent
            )

        else:

            print(
                "NONE - HUMAN SELECTION REQUIRED"
            )

        print(
            f"Suggestion source: "
            f"{suggestion_source}"
        )

        print(
            f"Suggested escalation: "
            f"{'TRUE' if escalation else 'FALSE'}"
        )

        # ----------------------------------------------------
        # DISPLAY AVAILABLE ACTIONS
        # ----------------------------------------------------

        print("\nENTER = accept suggestion")
        print("1-10 = choose intent")
        print("E = toggle escalation")
        print("S = skip")

        # ----------------------------------------------------
        # GET HUMAN DECISION
        # ----------------------------------------------------

        while True:

            action = input(
                "\nYour choice: "
            ).strip().lower()

            # ------------------------------------------------
            # ENTER
            # ------------------------------------------------
            #
            # ENTER is allowed only if an automatic suggestion
            # exists. This prevents accidental use of a fake
            # default label.
            # ------------------------------------------------

            if action == "":

                if suggested_intent is not None:

                    final_intent = (
                        suggested_intent
                    )

                    break

                else:

                    print(
                        "\nThere is no automatic suggestion."
                    )

                    print(
                        "Please choose an intent number "
                        "from 1 to 10."
                    )

                    # Stay in the current example.
                    continue

            # ------------------------------------------------
            # NUMBER 1-10
            # ------------------------------------------------

            elif action.isdigit():

                number = int(action)

                if 1 <= number <= 10:

                    final_intent = (
                        INTENTS[number - 1]
                    )

                    break

                print(
                    "Please enter a number from 1 to 10."
                )

            # ------------------------------------------------
            # E = TOGGLE ESCALATION
            # ------------------------------------------------

            elif action == "e":

                escalation = not escalation

                print(
                    f"Escalation is now "
                    f"{'TRUE' if escalation else 'FALSE'}"
                )

            # ------------------------------------------------
            # S = SKIP
            # ------------------------------------------------

            elif action == "s":

                print(
                    "Skipped. Progress will be saved."
                )

                # Save immediately before moving on.
                df.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                final_intent = None

                break

            # ------------------------------------------------
            # INVALID INPUT
            # ------------------------------------------------

            else:

                print(
                    "Invalid input."
                )

                print(
                    "Use ENTER, 1-10, E, or S."
                )

        # ----------------------------------------------------
        # SAVE HUMAN DECISION
        # ----------------------------------------------------

        if final_intent is not None:

            # Save the human-selected intent.
            df.at[
                row_number,
                "true_intent"
            ] = str(final_intent)

            # Save the final escalation decision.
            df.at[
                row_number,
                "true_escalation"
            ] = bool(escalation)

            # Record that the label was human verified.
            df.at[
                row_number,
                "human_review_note"
            ] = (
                "Human verified AI/rule pre-label; "
                "accepted or corrected."
            )

            print(
                f"\nSaved intent: "
                f"{final_intent}"
            )

            print(
                f"Saved escalation: "
                f"{'TRUE' if escalation else 'FALSE'}"
            )

        # ----------------------------------------------------
        # SAVE AFTER EVERY EXAMPLE
        # ----------------------------------------------------

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    completed = (
        df["true_intent"]
        .notna()
        .sum()
    )

    remaining = (
        total - completed
    )

    print("\n")
    print("=" * 70)
    print("HUMAN VERIFICATION STATUS")
    print("=" * 70)

    print(
        f"Total examples: {total}"
    )

    print(
        f"Human verified: {completed}"
    )

    print(
        f"Remaining: {remaining}"
    )

    print("\nSaved file:")
    print(OUTPUT_FILE)

    if remaining == 0:

        print(
            "\nAll 200 examples have been human verified."
        )

    else:

        print(
            "\nRun the script again to continue. "
            "Completed examples will be skipped."
        )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()