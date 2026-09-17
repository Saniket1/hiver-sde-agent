"""
AI-Assisted Golden Set Labeling
--------------------------------

Purpose:
    Use Gemini to create provisional intent and escalation labels
    for the 200-example golden-set candidate file.

Important:
    These AI labels are NOT the final golden-set labels.
    They must be human-verified before evaluation.

The script:
    1. Loads the 200 golden candidates.
    2. Sends each customer message to Gemini.
    3. Requests exactly one intent and one escalation decision.
    4. Handles Gemini rate limits (HTTP 429).
    5. Retries temporary failures.
    6. Saves progress after every example.
    7. Can resume from previous partial runs.
"""

# ------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

from pathlib import Path
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

# Resolve the project root dynamically.
# This makes the script work correctly regardless of the
# current PowerShell working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the .env file from the project root.
load_dotenv(PROJECT_ROOT / ".env")


# ------------------------------------------------------------
# GEMINI CONFIGURATION
# ------------------------------------------------------------

# Read the Gemini API key from the .env file.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Stop immediately if the API key is missing.
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )

# Create the current Google GenAI client.
client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini model that was successfully tested in this project.
MODEL_NAME = "gemini-3.6-flash"


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

# Input file containing the 200 golden-set candidates.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_candidates.csv"
)

# Main output file containing AI suggestions.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_ai_labeled.csv"
)

# File containing examples that still require human review.
REVIEW_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_needs_review.csv"
)

# Final golden-set placeholder.
# This file will contain the AI suggestions for now, but the
# true_intent and true_escalation columns remain for human
# verification.
FINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.csv"
)


# ------------------------------------------------------------
# RATE-LIMIT SETTINGS
# ------------------------------------------------------------

# Gemini Free Tier currently has a low requests-per-minute
# limit for this model in the user's current API setup.
#
# Waiting 15 seconds between successful requests keeps the
# request rate comfortably below 5 requests/minute.
REQUEST_DELAY_SECONDS = 15

# Maximum number of retries for a temporary API failure.
MAX_RETRIES = 5

# Initial wait after a rate-limit error.
INITIAL_RETRY_WAIT_SECONDS = 20


# ------------------------------------------------------------
# FROZEN INTENT TAXONOMY
# ------------------------------------------------------------

# These are the 10 intents already created from the actual
# AmazonHelp conversation clusters.
#
# IMPORTANT:
# Do not change these labels during golden-set generation,
# because the taxonomy has already been frozen for the project.

INTENTS = {
    "Order_Delivery_Tracking": (
        "Customer wants to know where an order or package is, "
        "its tracking status, or the expected arrival status."
    ),

    "Delivery_Delay_or_Missed_Date": (
        "Order is late, delayed, or has missed a promised or "
        "expected delivery date."
    ),

    "Order_Cancellation_or_Change": (
        "Customer wants to cancel or change an order, or reports "
        "an unexpected order cancellation."
    ),

    "Missing_Wrong_or_Damaged_Item": (
        "Customer reports a missing, wrong, incomplete, or "
        "damaged item/package."
    ),

    "Returns_Refunds_or_Replacements": (
        "Customer wants to return an item, receive a refund, "
        "or obtain a replacement."
    ),

    "Payment_or_Charge_Issue": (
        "Payment problem, unexpected charge, billing problem, "
        "cashback issue, or payment-related concern."
    ),

    "Account_Login_or_Security": (
        "Login, account access, authentication, verification, "
        "security, or unauthorized account activity."
    ),

    "Product_Availability_or_Pricing": (
        "Product availability, stock, price, promotion, "
        "discount, or sale-related question."
    ),

    "Product_or_Service_Information": (
        "General Amazon product or service information, "
        "compatibility, feature, or how-something-works question."
    ),

    "Prime_Video_Content_Availability": (
        "Question about availability, release, region, season, "
        "movie, show, or other Prime Video content."
    ),
}


# ------------------------------------------------------------
# HELPER FUNCTION: BUILD PROMPT
# ------------------------------------------------------------

def build_prompt(customer_message: str) -> str:
    """
    Build the classification prompt sent to Gemini.

    The model is explicitly instructed to:
        - choose exactly one intent
        - estimate confidence
        - decide escalation
        - provide a short reason
        - return JSON only
    """

    # Convert the frozen intent dictionary into readable text
    # for the Gemini prompt.
    intent_text = "\n".join(
        f"- {intent}: {description}"
        for intent, description in INTENTS.items()
    )

    prompt = f"""
You are classifying a customer-support message for AmazonHelp.

Choose EXACTLY ONE intent from the allowed taxonomy below.

ALLOWED INTENTS:
{intent_text}

ESCALATION RULE:
Set escalation=true when the customer appears to need a human
agent because of issues such as:
- unresolved problems
- repeated failed support attempts
- serious complaint
- fraud/security concern
- unusual or complex case
- request for human intervention
- emotionally escalated situation

Set escalation=false when the request is routine and could reasonably
be handled automatically.

CUSTOMER MESSAGE:
{customer_message}

Return ONLY valid JSON in exactly this structure:

{{
  "intent": "ONE_ALLOWED_INTENT",
  "confidence": 0.0,
  "escalation": false,
  "reason": "short explanation"
}}

Requirements:
- intent must exactly match one of the allowed intent names.
- confidence must be a number between 0 and 1.
- escalation must be true or false.
- reason should be concise.
- Do not use Markdown.
- Do not include ```json.
- Do not include any text outside the JSON object.
"""

    return prompt.strip()


# ------------------------------------------------------------
# HELPER FUNCTION: EXTRACT JSON
# ------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from Gemini's response.

    Gemini should return JSON only, but this function is defensive:
    it also handles responses surrounded by Markdown code fences
    or extra whitespace.
    """

    # Remove leading/trailing whitespace.
    cleaned = text.strip()

    # Remove a possible Markdown JSON code fence.
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Find the first JSON object.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError(
            f"Could not find a JSON object in Gemini response: {text}"
        )

    json_text = match.group(0)

    # Convert the JSON string into a Python dictionary.
    result = json.loads(json_text)

    if not isinstance(result, dict):
        raise ValueError("Gemini response JSON was not an object.")

    return result


# ------------------------------------------------------------
# HELPER FUNCTION: VALIDATE MODEL OUTPUT
# ------------------------------------------------------------

def validate_result(result: dict) -> dict:
    """
    Validate Gemini's classification result.

    This prevents malformed responses from being accepted as
    valid golden-set labels.
    """

    # Required fields.
    required_fields = {
        "intent",
        "confidence",
        "escalation",
        "reason",
    }

    missing = required_fields - set(result.keys())

    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )

    # Validate intent.
    intent = str(result["intent"]).strip()

    if intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned by Gemini: {intent}"
        )

    # Validate confidence.
    confidence = float(result["confidence"])

    if confidence < 0 or confidence > 1:
        raise ValueError(
            f"Confidence must be between 0 and 1, got {confidence}"
        )

    # Validate escalation.
    escalation = result["escalation"]

    # Accept true/false as well as boolean-like strings.
    if isinstance(escalation, str):
        escalation_lower = escalation.strip().lower()

        if escalation_lower == "true":
            escalation = True
        elif escalation_lower == "false":
            escalation = False
        else:
            raise ValueError(
                f"Invalid escalation value: {escalation}"
            )

    elif not isinstance(escalation, bool):
        raise ValueError(
            f"Escalation must be boolean, got {type(escalation)}"
        )

    # Return a clean normalized result.
    return {
        "intent": intent,
        "confidence": confidence,
        "escalation": bool(escalation),
        "reason": str(result["reason"]).strip(),
    }


# ------------------------------------------------------------
# HELPER FUNCTION: CALL GEMINI WITH RETRIES
# ------------------------------------------------------------

def classify_message(customer_message: str) -> dict:
    """
    Send one customer message to Gemini.

    Handles:
        - rate-limit errors
        - temporary API failures
        - malformed model responses

    Returns:
        A dictionary containing either a successful classification
        or an error field.
    """

    prompt = build_prompt(customer_message)

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            # Send the request to Gemini.
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            # Make sure Gemini returned actual text.
            response_text = getattr(response, "text", None)

            if not response_text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            # Parse Gemini JSON.
            parsed = extract_json(response_text)

            # Validate the parsed result.
            validated = validate_result(parsed)

            # Add the raw Gemini response for debugging/auditability.
            validated["raw_response"] = response_text

            return validated

        except Exception as exc:

            error_text = str(exc)

            # ------------------------------------------------
            # RATE LIMIT ERROR
            # ------------------------------------------------
            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                # Use an increasing wait time after each retry.
                wait_seconds = (
                    INITIAL_RETRY_WAIT_SECONDS * attempt
                )

                print(
                    f"  Rate limit reached. "
                    f"Waiting {wait_seconds}s "
                    f"before retry {attempt}/{MAX_RETRIES}..."
                )

                time.sleep(wait_seconds)

                continue

            # ------------------------------------------------
            # OTHER TEMPORARY/PROCESSING ERROR
            # ------------------------------------------------
            print(
                f"  Gemini error on attempt "
                f"{attempt}/{MAX_RETRIES}: {error_text}"
            )

            # Wait briefly before another attempt.
            time.sleep(5)

    # --------------------------------------------------------
    # ALL RETRIES FAILED
    # --------------------------------------------------------

    return {
        "error": (
            "Gemini classification failed after "
            f"{MAX_RETRIES} attempts."
        )
    }


# ------------------------------------------------------------
# HELPER FUNCTION: LOAD EXISTING PROGRESS
# ------------------------------------------------------------

def load_existing_results() -> pd.DataFrame | None:
    """
    Load the previous AI-labeled file when it exists.

    This allows the script to resume rather than starting from
    zero after an interruption.
    """

    if not OUTPUT_FILE.exists():
        return None

    try:
        existing = pd.read_csv(OUTPUT_FILE)

        print(
            f"Existing output found: {len(existing)} rows"
        )

        return existing

    except Exception as exc:
        print(
            "Could not read existing output. "
            f"Starting fresh. Error: {exc}"
        )

        return None


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("AI-ASSISTED GOLDEN SET LABELING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD INPUT DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    candidates = pd.read_csv(INPUT_FILE)

    print(f"Input candidates: {len(candidates)}")

    # --------------------------------------------------------
    # CHECK REQUIRED INPUT COLUMNS
    # --------------------------------------------------------

    required_columns = {
        "golden_candidate_id",
        "conversation_id",
        "customer_message",
        "agent_response",
        "timestamp",
    }

    missing_columns = (
        required_columns - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # LOAD PREVIOUS RESULTS
    # --------------------------------------------------------

    existing = load_existing_results()

    # Start with an empty results table if no previous run exists.
    if existing is None:

        results = candidates.copy()

        # Create columns that will be populated by Gemini.
        results["ai_suggested_intent"] = pd.NA
        results["ai_confidence"] = pd.NA
        results["ai_suggested_escalation"] = pd.NA
        results["ai_reason"] = pd.NA
        results["needs_human_review"] = True
        results["true_intent"] = pd.NA
        results["true_escalation"] = pd.NA
        results["human_review_note"] = pd.NA
        results["llm_error"] = pd.NA

    else:

        # Use the previous file so completed results are preserved.
        results = existing.copy()

        # Make sure newly expected columns exist.
        for column in [
            "ai_suggested_intent",
            "ai_confidence",
            "ai_suggested_escalation",
            "ai_reason",
            "needs_human_review",
            "true_intent",
            "true_escalation",
            "human_review_note",
            "llm_error",
        ]:

            if column not in results.columns:
                results[column] = pd.NA

    # --------------------------------------------------------
    # CREATE LOOKUP FOR PREVIOUS RESULTS
    # --------------------------------------------------------

    # Map each candidate ID to its current row number.
    row_lookup = {
        str(value): index
        for index, value in enumerate(
            results["golden_candidate_id"]
        )
    }

    # --------------------------------------------------------
    # PROCESS EACH CANDIDATE
    # --------------------------------------------------------

    total = len(candidates)

    for position, (_, candidate) in enumerate(
        candidates.iterrows(),
        start=1,
    ):

        candidate_id = str(
            candidate["golden_candidate_id"]
        )

        # Find corresponding row in the output dataframe.
        result_index = row_lookup.get(candidate_id)

        if result_index is None:
            continue

        # ----------------------------------------------------
        # RESUME LOGIC
        # ----------------------------------------------------
        #
        # If this row already has a valid AI intent and does
        # not have an API error, do not send it again.
        #
        # This is important because the Gemini Free Tier has
        # strict request limits.
        # ----------------------------------------------------

        existing_intent = results.at[
            result_index,
            "ai_suggested_intent",
        ]

        existing_error = results.at[
            result_index,
            "llm_error",
        ]

        already_completed = (
            pd.notna(existing_intent)
            and str(existing_intent).strip() in INTENTS
            and (
                pd.isna(existing_error)
                or str(existing_error).strip() == ""
            )
        )

        if already_completed:

            print(
                f"[{position}/{total}] "
                f"Already completed - skipping."
            )

            continue

        # ----------------------------------------------------
        # GET CUSTOMER MESSAGE
        # ----------------------------------------------------

        customer_message = str(
            candidate["customer_message"]
        ).strip()

        print(
            f"\n[{position}/{total}] "
            f"Processing candidate {candidate_id}"
        )

        print(
            f"  Message: {customer_message[:150]}"
        )

        # ----------------------------------------------------
        # CALL GEMINI
        # ----------------------------------------------------

        result = classify_message(customer_message)

        # ----------------------------------------------------
        # HANDLE SUCCESSFUL CLASSIFICATION
        # ----------------------------------------------------

        if "error" not in result:

            results.at[
                result_index,
                "ai_suggested_intent",
            ] = result["intent"]

            results.at[
                result_index,
                "ai_confidence",
            ] = result["confidence"]

            results.at[
                result_index,
                "ai_suggested_escalation",
            ] = result["escalation"]

            results.at[
                result_index,
                "ai_reason",
            ] = result["reason"]

            # AI labels with confidence >= 0.80 can initially
            # be treated as higher-confidence suggestions.
            #
            # IMPORTANT:
            # They are STILL NOT final human labels.
            needs_review = result["confidence"] < 0.80

            results.at[
                result_index,
                "needs_human_review",
            ] = needs_review

            # Clear any previous error from an earlier failed run.
            results.at[
                result_index,
                "llm_error",
            ] = pd.NA

            print(
                f"  Intent: {result['intent']}"
            )

            print(
                f"  Confidence: "
                f"{result['confidence']:.2f}"
            )

            print(
                f"  Escalation: "
                f"{result['escalation']}"
            )

        # ----------------------------------------------------
        # HANDLE FAILED CLASSIFICATION
        # ----------------------------------------------------

        else:

            results.at[
                result_index,
                "ai_suggested_intent",
            ] = pd.NA

            results.at[
                result_index,
                "ai_confidence",
            ] = 0.0

            results.at[
                result_index,
                "ai_suggested_escalation",
            ] = False

            results.at[
                result_index,
                "ai_reason",
            ] = pd.NA

            results.at[
                result_index,
                "needs_human_review",
            ] = True

            results.at[
                result_index,
                "llm_error",
            ] = result["error"]

            print(
                f"  FAILED: {result['error']}"
            )

        # ----------------------------------------------------
        # SAVE AFTER EVERY EXAMPLE
        # ----------------------------------------------------
        #
        # This protects the work if:
        # - the terminal is closed
        # - the API quota is exhausted
        # - the computer restarts
        # - the script is interrupted
        # ----------------------------------------------------

        results.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        # ----------------------------------------------------
        # RATE-LIMIT PROTECTION
        # ----------------------------------------------------
        #
        # Wait after every successful API request.
        # This keeps us below the Free Tier request rate.
        # ----------------------------------------------------

        if "error" not in result:

            print(
                f"  Waiting "
                f"{REQUEST_DELAY_SECONDS}s "
                f"before the next API request..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # --------------------------------------------------------
    # CREATE HUMAN-REVIEW FILE
    # --------------------------------------------------------

    review_mask = (
        results["needs_human_review"]
        .fillna(True)
        .astype(bool)
    )

    review = results.loc[
        review_mask
    ].copy()

    review.to_csv(
        REVIEW_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # CREATE CURRENT GOLDEN SET FILE
    # --------------------------------------------------------
    #
    # This is a working file at this stage.
    # true_intent and true_escalation remain blank until
    # human verification is completed.
    # --------------------------------------------------------

    results.to_csv(
        FINAL_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    valid_ai = results[
        results["ai_suggested_intent"].notna()
    ]

    high_confidence = valid_ai[
        pd.to_numeric(
            valid_ai["ai_confidence"],
            errors="coerce",
        ) >= 0.80
    ]

    print("\n")
    print("=" * 70)
    print("AI-ASSISTED LABELING COMPLETE")
    print("=" * 70)

    print(
        f"Total candidates: {len(results)}"
    )

    print(
        f"Valid AI suggestions: {len(valid_ai)}"
    )

    print(
        f"High-confidence AI suggestions: "
        f"{len(high_confidence)}"
    )

    print(
        f"Requires human review: "
        f"{len(review)}"
    )

    print("\nAI-labeled file:")
    print(OUTPUT_FILE)

    print("\nReview file:")
    print(REVIEW_FILE)

    print("\nWorking golden-set file:")
    print(FINAL_FILE)

    print("\nIMPORTANT:")
    print(
        "AI labels remain provisional until "
        "human verification."
    )


# ------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()