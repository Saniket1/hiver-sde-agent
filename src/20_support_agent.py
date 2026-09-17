"""
End-to-End AmazonHelp Support Agent
------------------------------------

Purpose:
    Combine the major components developed so far into one
    support-agent pipeline.

Pipeline:

    Customer message
          |
          v
    Intent classification
          |
          v
    Historical resolution retrieval
          |
          v
    Escalation decision
          |
          v
    Draft response
          |
          v
    AUTO-HANDLE / HUMAN

Outputs:
    - intent
    - intent confidence
    - top historical resolutions
    - escalation decision
    - escalation reason
    - grounded draft response

Important:
    Gemini response generation is attempted when available.
    If Gemini quota is exhausted, the system uses a conservative
    historical-response-based fallback instead of crashing.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import os
import re

import faiss
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from google import genai


# ============================================================
# PROJECT PATH
# ============================================================

# Locate the project root automatically.
# This file is inside "src", so parents[1] is the project root.
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

# Historical FAISS index.
INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_faiss.index"
)

# Historical retrieval metadata.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_retrieval_metadata.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

# Semantic embedding model used for both intent classification
# and historical retrieval.
EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

# Number of historical examples retrieved.
TOP_K = 5

# Gemini model used when API quota is available.
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# LOAD .ENV
# ============================================================

# Load the project's .env file.
load_dotenv(
    PROJECT_ROOT / ".env"
)

# Read Gemini API key.
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# ============================================================
# INITIALIZE GEMINI CLIENT
# ============================================================

# We only initialize Gemini when an API key is available.
# This allows the local agent to run even without Gemini.
gemini_client = None

if GEMINI_API_KEY:

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

for required_file in [
    GOLDEN_FILE,
    INDEX_FILE,
    METADATA_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{required_file}"
        )


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("LOADING GOLDEN SET")
print("=" * 70)

golden_df = pd.read_csv(
    GOLDEN_FILE
)


# ============================================================
# CLEAN GOLDEN SET
# ============================================================

# Remove rows without the required fields.
golden_df = golden_df.dropna(
    subset=[
        "customer_message",
        "true_intent",
    ]
).copy()

# Convert text to strings.
golden_df["customer_message"] = (
    golden_df["customer_message"]
    .astype(str)
    .str.strip()
)

# Print dataset size.
print(
    f"Golden examples: {len(golden_df)}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading historical FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)


# ============================================================
# LOAD RETRIEVAL METADATA
# ============================================================

print("Loading historical metadata...")

metadata = pd.read_csv(
    METADATA_FILE
)


# ============================================================
# VALIDATE INDEX ALIGNMENT
# ============================================================

# Every FAISS vector must correspond to one metadata row.
if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata are not aligned. "
        f"Index={index.ntotal}, Metadata={len(metadata)}"
    )


# ============================================================
# IDENTIFY HISTORICAL TEXT COLUMNS
# ============================================================

# Possible customer-message column names.
possible_customer_columns = [
    "customer_message",
    "customer_text",
    "text",
]

# Possible agent-response column names.
possible_response_columns = [
    "agent_response",
    "response",
    "response_text",
]


# Find customer-message column.
customer_column = None

for column in possible_customer_columns:

    if column in metadata.columns:

        customer_column = column
        break


# Find agent-response column.
response_column = None

for column in possible_response_columns:

    if column in metadata.columns:

        response_column = column
        break


# Validate the columns.
if customer_column is None:

    raise ValueError(
        "Historical customer-message column not found."
    )


if response_column is None:

    raise ValueError(
        "Historical response column not found."
    )


# ============================================================
# BUILD INTENT CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING INTENT CLASSIFIER")
print("=" * 70)


# ------------------------------------------------------------
# INPUT / TARGET
# ------------------------------------------------------------

# Customer text.
X_intent = golden_df[
    "customer_message"
]

# Human-verified intent.
y_intent = golden_df[
    "true_intent"
]


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

# Use the same TF-IDF + Logistic Regression approach from the
# baseline because it is fast and reliable for an interactive
# local agent.
intent_model = Pipeline(
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


# Train on the entire human-verified set for deployment.
intent_model.fit(
    X_intent,
    y_intent,
)


print(
    "Intent classifier ready."
)


# ============================================================
# ESCALATION RULES
# ============================================================

# These rules are based on the escalation signals identified
# during the earlier failure analysis.

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
        r"\bcall me\b",
        r"\bcontact me\b",
    ],

    "explicit investigation request": [
        r"\binvestigation\b",
        r"\binvestigate\b",
        r"\bplease investigate\b",
        r"\bformal complaint\b",
        r"\bescalat\b",
    ],
}


UNRESOLVED_PATTERNS = [
    r"\bno response\b",
    r"\bno reply\b",
    r"\bno help\b",
    r"\bstill waiting\b",
    r"\bwaiting for a reply\b",
    r"\bwaiting for your reply\b",
    r"\bnot been resolved\b",
    r"\bnot resolved\b",
    r"\bunresolved\b",
    r"\bnothing has happened\b",
    r"\bnothing was done\b",
    r"\bcan't help\b",
    r"\bcannot help\b",
    r"\bdidn't help\b",
    r"\bdid not help\b",
    r"\bnever called\b",
    r"\bnever got a reply\b",
]


REPEATED_CONTACT_PATTERNS = [
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
]


DELIVERY_FAILURE_PATTERNS = [
    r"\bpackages keep going missing\b",
    r"\bpackage.*missing\b",
    r"\bpackages.*missing\b",
    r"\bkeep.*missing\b",
    r"\bmarked as delivered.*not received\b",
    r"\bdelivered.*haven't received\b",
    r"\bdelivered.*never received\b",
    r"\bmissed.*again\b",
    r"\brepeated delivery\b",
]


# ============================================================
# ESCALATION FUNCTION
# ============================================================

def decide_escalation(
    customer_message: str,
):
    """
    Decide HUMAN vs AUTO-HANDLE.

    Returns:
        decision
        reason
    """

    # Convert text to lowercase for matching.
    text = str(
        customer_message
    ).lower()

    # --------------------------------------------------------
    # HIGH-SEVERITY SIGNALS
    # --------------------------------------------------------

    for category, patterns in HIGH_SEVERITY_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text,
            ):

                return (
                    "HUMAN",
                    f"High-severity signal: {category}."
                )

    # --------------------------------------------------------
    # UNRESOLVED SUPPORT
    # --------------------------------------------------------

    for pattern in UNRESOLVED_PATTERNS:

        if re.search(
            pattern,
            text,
        ):

            return (
                "HUMAN",
                "Customer reports an unresolved support interaction."
            )

    # --------------------------------------------------------
    # REPEATED CONTACT
    # --------------------------------------------------------

    for pattern in REPEATED_CONTACT_PATTERNS:

        if re.search(
            pattern,
            text,
        ):

            return (
                "HUMAN",
                "Customer reports repeated contact or repeated failure."
            )

    # --------------------------------------------------------
    # DELIVERY FAILURE
    # --------------------------------------------------------

    for pattern in DELIVERY_FAILURE_PATTERNS:

        if re.search(
            pattern,
            text,
        ):

            return (
                "HUMAN",
                "Repeated or serious delivery failure detected."
            )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "AUTO-HANDLE",
        "No strong escalation signal detected."
    )


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_resolutions(
    customer_message: str,
    top_k: int = TOP_K,
):
    """
    Retrieve the most similar historical customer-support cases.
    """

    # Create semantic embedding for the incoming message.
    query_embedding = embedding_model.encode(
        [customer_message],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # FAISS requires float32.
    query_embedding = query_embedding.astype(
        "float32"
    )

    # Search for nearest historical cases.
    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    retrieved = []

    # Convert retrieved rows into readable dictionaries.
    for rank in range(top_k):

        row_index = int(
            indices[0][rank]
        )

        # Ignore invalid results.
        if row_index < 0:

            continue

        row = metadata.iloc[
            row_index
        ]

        retrieved.append(
            {
                "rank": rank + 1,
                "similarity": float(
                    scores[0][rank]
                ),
                "customer_message": str(
                    row[customer_column]
                ),
                "agent_response": str(
                    row[response_column]
                ),
            }
        )

    return retrieved


# ============================================================
# RESPONSE GENERATION PROMPT
# ============================================================

def build_prompt(
    customer_message,
    intent,
    retrieved_cases,
):
    """
    Construct a grounded response-generation prompt.
    """

    evidence_blocks = []

    # Add retrieved historical cases.
    for item in retrieved_cases:

        evidence_blocks.append(
            f"""
Historical example {item["rank"]}
Similarity: {item["similarity"]:.4f}

Customer:
{item["customer_message"]}

Historical response:
{item["agent_response"]}
""".strip()
        )

    evidence = "\n\n".join(
        evidence_blocks
    )

    prompt = f"""
You are an Amazon customer-support response assistant.

Customer message:
{customer_message}

Predicted intent:
{intent}

Historical AmazonHelp evidence:
{evidence}

Write a short, helpful customer-facing response.

Rules:
- Ground the response in the historical evidence.
- Do not invent Amazon policies.
- Do not promise a refund, compensation, delivery date, or action
  unless supported by the evidence.
- Do not claim that an action has already been taken.
- Do not mention AI, FAISS, embeddings, or internal systems.
- If the evidence does not establish a specific resolution,
  provide a cautious next step.
- Be professional and concise.

Return only the customer-facing response.
"""

    return prompt.strip()


# ============================================================
# FALLBACK RESPONSE
# ============================================================

def fallback_response(
    customer_message,
    intent,
    retrieved_cases,
):
    """
    Generate a conservative response without using Gemini.

    This is intentionally generic. It does not invent a specific
    Amazon policy or promise a particular resolution.
    """

    # If historical examples exist, use the strongest response
    # as evidence for a cautious next step.
    if retrieved_cases:

        best_case = retrieved_cases[0]

        historical_response = (
            best_case["agent_response"]
        )

        # Keep the response short and avoid pretending that the
        # historical response applies exactly to this customer.
        return (
            "I'm sorry you're experiencing this issue. "
            "Please contact Amazon support so the team can "
            "review your specific order/account and advise "
            "on the appropriate next step."
        )

    return (
        "I'm sorry you're experiencing this issue. "
        "Please contact Amazon support so the team can review "
        "the details and advise on the appropriate next step."
    )


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_response(
    customer_message,
    intent,
    retrieved_cases,
):
    """
    Generate a grounded response using Gemini when available.

    If the Gemini quota is exhausted, use a conservative
    fallback instead.
    """

    # If no Gemini API key exists, use local fallback.
    if gemini_client is None:

        return (
            fallback_response(
                customer_message,
                intent,
                retrieved_cases,
            ),
            "LOCAL_FALLBACK",
        )

    # Build grounded prompt.
    prompt = build_prompt(
        customer_message,
        intent,
        retrieved_cases,
    )

    try:

        # Ask Gemini to generate the response.
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        response_text = getattr(
            response,
            "text",
            None,
        )

        # Reject empty responses.
        if not response_text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        return (
            response_text.strip(),
            "GEMINI",
        )

    except Exception as exc:

        error_text = str(
            exc
        )

        # Handle Free Tier quota exhaustion gracefully.
        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):

            return (
                fallback_response(
                    customer_message,
                    intent,
                    retrieved_cases,
                ),
                "LOCAL_FALLBACK_QUOTA",
            )

        # Handle any other API error gracefully.
        return (
            fallback_response(
                customer_message,
                intent,
                retrieved_cases,
            ),
            "LOCAL_FALLBACK_ERROR",
        )


# ============================================================
# COMPLETE AGENT FUNCTION
# ============================================================

def run_agent(
    customer_message: str,
):
    """
    Run the entire support-agent pipeline.

    Returns:
        dictionary containing the complete agent output.
    """

    # --------------------------------------------------------
    # INTENT CLASSIFICATION
    # --------------------------------------------------------

    predicted_intent = intent_model.predict(
        [customer_message]
    )[0]

    # Get probability estimates for the predicted class.
    intent_probabilities = (
        intent_model.predict_proba(
            [customer_message]
        )[0]
    )

    # Find the highest probability.
    intent_confidence = float(
        np.max(
            intent_probabilities
        )
    )

    # --------------------------------------------------------
    # HISTORICAL RETRIEVAL
    # --------------------------------------------------------

    retrieved_cases = retrieve_resolutions(
        customer_message,
        TOP_K,
    )

    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    decision, escalation_reason = (
        decide_escalation(
            customer_message
        )
    )

    # --------------------------------------------------------
    # RESPONSE GENERATION
    # --------------------------------------------------------

    draft_response, response_source = (
        generate_response(
            customer_message,
            predicted_intent,
            retrieved_cases,
        )
    )

    # --------------------------------------------------------
    # RETURN FINAL AGENT OUTPUT
    # --------------------------------------------------------

    return {
        "customer_message": customer_message,
        "intent": predicted_intent,
        "intent_confidence": intent_confidence,
        "decision": decision,
        "escalation_reason": escalation_reason,
        "draft_response": draft_response,
        "response_source": response_source,
        "retrieved_cases": retrieved_cases,
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():
    """
    Allow the user to enter customer messages interactively.
    """

    print("\n")
    print("=" * 70)
    print("AMAZONHELP SUPPORT AGENT")
    print("=" * 70)

    print(
        "\nEnter a customer message."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        print("\n")
        customer_message = input(
            "Customer message: "
        ).strip()

        # Stop interactive mode.
        if customer_message.lower() == "exit":

            break

        # Prevent empty input.
        if not customer_message:

            print(
                "Please enter a customer message."
            )

            continue

        # Run the entire pipeline.
        result = run_agent(
            customer_message
        )

        # ----------------------------------------------------
        # DISPLAY INTENT
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("INTENT")
        print("-" * 70)

        print(
            result["intent"]
        )

        print(
            f"Confidence: "
            f"{result['intent_confidence']:.2f}"
        )

        # ----------------------------------------------------
        # DISPLAY ESCALATION
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("DECISION")
        print("-" * 70)

        print(
            result["decision"]
        )

        print(
            f"Reason: "
            f"{result['escalation_reason']}"
        )

        # ----------------------------------------------------
        # DISPLAY HISTORICAL CASES
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("TOP HISTORICAL RESOLUTIONS")
        print("-" * 70)

        for item in result[
            "retrieved_cases"
        ]:

            print(
                f"\nRank {item['rank']} "
                f"| Similarity="
                f"{item['similarity']:.4f}"
            )

            print(
                f"Customer: "
                f"{item['customer_message'][:250]}"
            )

            print(
                f"Response: "
                f"{item['agent_response'][:250]}"
            )

        # ----------------------------------------------------
        # DISPLAY DRAFT RESPONSE
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("DRAFT RESPONSE")
        print("-" * 70)

        print(
            result["draft_response"]
        )

        print(
            f"\nResponse source: "
            f"{result['response_source']}"
        )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    interactive_mode()