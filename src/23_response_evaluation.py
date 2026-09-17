"""
Response Evaluation Preparation
--------------------------------

Purpose:
    Prepare and perform local quality checks for the support-agent
    responses before running an LLM-as-judge evaluation.

What this script checks:

    1. Response exists.
    2. Response is not excessively long.
    3. Response does not expose internal implementation details.
    4. Response does not invent obvious unsupported promises.
    5. Response is reasonably grounded in retrieved evidence.
    6. Prepare a JSONL file that can later be sent to an LLM judge.

Important:
    These are heuristic checks, NOT a substitute for the
    assignment's LLM-as-judge evaluation.

    Gemini is NOT called by this script because the current API
    quota has been exhausted.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json
import re

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT / OUTPUT FILES
# ============================================================

# The final human-verified golden set.
GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

# Detailed intent/retrieval evaluation results.
EVALUATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_detailed.csv"
)

# Output file containing prepared response-evaluation records.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "response_evaluation.csv"
)

# JSONL file prepared for later LLM-as-judge evaluation.
JUDGE_FILE = (
    PROJECT_ROOT
    / "results"
    / "response_llm_judge_input.jsonl"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

# Make sure the results directory exists.
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


if not EVALUATION_FILE.exists():

    raise FileNotFoundError(
        f"Evaluation file not found:\n{EVALUATION_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RESPONSE EVALUATION PREPARATION")
print("=" * 70)

# Load the golden examples.
golden_df = pd.read_csv(
    GOLDEN_FILE
)

# Load the local evaluation details.
evaluation_df = pd.read_csv(
    EVALUATION_FILE
)

print(
    f"Golden examples: {len(golden_df)}"
)

print(
    f"Evaluation examples: {len(evaluation_df)}"
)


# ============================================================
# REQUIRED GOLDEN-SET COLUMNS
# ============================================================

required_golden_columns = {
    "customer_message",
    "true_intent",
    "true_escalation",
}

missing_golden_columns = (
    required_golden_columns
    - set(golden_df.columns)
)

if missing_golden_columns:

    raise ValueError(
        "Golden set is missing columns: "
        f"{sorted(missing_golden_columns)}"
    )


# ============================================================
# REQUIRED EVALUATION COLUMNS
# ============================================================

required_evaluation_columns = {
    "customer_message",
    "true_intent",
    "predicted_intent",
    "top1_similarity",
    "avg_top5_similarity",
}

missing_evaluation_columns = (
    required_evaluation_columns
    - set(evaluation_df.columns)
)

if missing_evaluation_columns:

    raise ValueError(
        "Evaluation file is missing columns: "
        f"{sorted(missing_evaluation_columns)}"
    )


# ============================================================
# CLEAN CUSTOMER MESSAGES
# ============================================================

golden_df["customer_message"] = (
    golden_df["customer_message"]
    .astype(str)
    .str.strip()
)

evaluation_df["customer_message"] = (
    evaluation_df["customer_message"]
    .astype(str)
    .str.strip()
)


# ============================================================
# MERGE DATA
# ============================================================

# The detailed evaluation contains one row per held-out
# evaluation example. We merge the human escalation label
# from the golden set.

golden_lookup = golden_df[
    [
        "customer_message",
        "true_escalation",
    ]
].copy()


response_df = evaluation_df.merge(
    golden_lookup,
    on="customer_message",
    how="left",
)


# ============================================================
# RESPONSE TEXT SOURCE
# ============================================================

# IMPORTANT:
#
# The current evaluation_detailed.csv does not contain generated
# responses because the local evaluation harness deliberately
# avoided Gemini.
#
# Therefore, we create a conservative placeholder that makes
# the missing response state explicit.
#
# Later, once Gemini quota is available, the actual generated
# responses can be inserted into this same evaluation structure.

response_df["draft_response"] = (
    "RESPONSE_NOT_AVAILABLE_FOR_LOCAL_EVALUATION"
)


# ============================================================
# BASIC RESPONSE CHECK
# ============================================================

def check_response_exists(
    response: str,
):
    """
    Check whether a usable response exists.
    """

    text = str(
        response
    ).strip()

    if not text:
        return False

    if (
        text == ""
        or text == "nan"
        or "RESPONSE_NOT_AVAILABLE" in text
    ):
        return False

    return True


# ============================================================
# INTERNAL-DETAIL CHECK
# ============================================================

# These terms should never appear in a customer-facing response.
INTERNAL_TERMS = [
    "faiss",
    "embedding",
    "sentence-transformer",
    "tf-idf",
    "logistic regression",
    "vector index",
    "similarity score",
    "llm",
    "language model",
]


def contains_internal_details(
    response: str,
):
    """
    Detect internal implementation terminology.
    """

    text = str(
        response
    ).lower()

    matched = [
        term
        for term in INTERNAL_TERMS
        if term in text
    ]

    return (
        len(matched) > 0,
        matched,
    )


# ============================================================
# UNSUPPORTED PROMISE CHECK
# ============================================================

# We flag strong promises because historically grounded support
# responses should not invent actions or guarantees.

PROMISE_PATTERNS = [
    r"\bwe will refund\b",
    r"\byou will receive a refund\b",
    r"\byou will definitely receive\b",
    r"\bwe guarantee\b",
    r"\bguaranteed delivery\b",
    r"\bwe have refunded\b",
    r"\bwe have issued\b",
    r"\bwe have contacted\b",
    r"\bwe will contact you\b",
    r"\byour order will arrive\b",
]


def contains_unsupported_promise(
    response: str,
):
    """
    Flag language that may represent an unsupported promise.
    """

    text = str(
        response
    ).lower()

    matches = []

    for pattern in PROMISE_PATTERNS:

        if re.search(
            pattern,
            text,
        ):

            matches.append(
                pattern
            )

    return (
        len(matches) > 0,
        matches,
    )


# ============================================================
# OVERLY-LONG RESPONSE CHECK
# ============================================================

def is_too_long(
    response: str,
    max_words: int = 120,
):
    """
    Check whether a response is unnecessarily long.

    Support responses should generally be concise.
    """

    word_count = len(
        str(response)
        .split()
    )

    return (
        word_count > max_words,
        word_count,
    )


# ============================================================
# EVIDENCE OVERLAP CHECK
# ============================================================

def evidence_overlap(
    customer_message: str,
    response: str,
):
    """
    Calculate a lightweight lexical overlap score.

    This is NOT a semantic grounding score.

    It is only a simple diagnostic that checks whether meaningful
    words from the customer message appear in the response.
    """

    # Normalize words.
    customer_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            str(customer_message).lower(),
        )
    )

    response_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            str(response).lower(),
        )
    )

    # Ignore generic stop words.
    stop_words = {
        "this",
        "that",
        "with",
        "have",
        "your",
        "from",
        "they",
        "what",
        "when",
        "were",
        "been",
        "just",
        "will",
        "would",
        "could",
        "about",
        "there",
        "their",
        "them",
        "then",
        "than",
        "into",
        "please",
        "amazon",
        "customer",
    }

    customer_words = {
        word
        for word in customer_words
        if word not in stop_words
    }

    # Avoid division by zero.
    if not customer_words:

        return 0.0

    overlap = (
        customer_words
        & response_words
    )

    return (
        len(overlap)
        / len(customer_words)
    )


# ============================================================
# RUN LOCAL CHECKS
# ============================================================

response_exists = []
internal_detail_flags = []
internal_detail_terms = []
promise_flags = []
promise_terms = []
too_long_flags = []
word_counts = []
overlap_scores = []


for _, row in response_df.iterrows():

    # Read current draft response.
    response = row[
        "draft_response"
    ]

    # --------------------------------------------------------
    # RESPONSE EXISTS
    # --------------------------------------------------------

    response_exists.append(
        check_response_exists(
            response
        )
    )

    # --------------------------------------------------------
    # INTERNAL TERMINOLOGY
    # --------------------------------------------------------

    internal_flag, internal_terms = (
        contains_internal_details(
            response
        )
    )

    internal_detail_flags.append(
        internal_flag
    )

    internal_detail_terms.append(
        ", ".join(
            internal_terms
        )
    )

    # --------------------------------------------------------
    # UNSUPPORTED PROMISES
    # --------------------------------------------------------

    promise_flag, promise_matches = (
        contains_unsupported_promise(
            response
        )
    )

    promise_flags.append(
        promise_flag
    )

    promise_terms.append(
        ", ".join(
            promise_matches
        )
    )

    # --------------------------------------------------------
    # RESPONSE LENGTH
    # --------------------------------------------------------

    long_flag, count = (
        is_too_long(
            response
        )
    )

    too_long_flags.append(
        long_flag
    )

    word_counts.append(
        count
    )

    # --------------------------------------------------------
    # LEXICAL OVERLAP
    # --------------------------------------------------------

    overlap_scores.append(
        evidence_overlap(
            row["customer_message"],
            response,
        )
    )


# ============================================================
# ADD CHECK RESULTS
# ============================================================

response_df[
    "response_exists"
] = response_exists

response_df[
    "contains_internal_details"
] = internal_detail_flags

response_df[
    "internal_terms"
] = internal_detail_terms

response_df[
    "contains_unsupported_promise"
] = promise_flags

response_df[
    "promise_terms"
] = promise_terms

response_df[
    "too_long"
] = too_long_flags

response_df[
    "response_word_count"
] = word_counts

response_df[
    "customer_response_word_overlap"
] = overlap_scores


# ============================================================
# OVERALL RESPONSE SAFETY FLAG
# ============================================================

# A response is locally "clean" when:
#
#     - a response exists
#     - no internal terms occur
#     - no obvious unsupported promise occurs
#     - response isn't excessively long
#
# This does NOT mean the response is objectively good.
response_df[
    "local_quality_pass"
] = (
    response_df[
        "response_exists"
    ]
    &
    ~response_df[
        "contains_internal_details"
    ]
    &
    ~response_df[
        "contains_unsupported_promise"
    ]
    &
    ~response_df[
        "too_long"
    ]
)


# ============================================================
# SAVE LOCAL EVALUATION
# ============================================================

response_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CREATE LLM-AS-JUDGE INPUT
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING LLM-AS-JUDGE INPUT")
print("=" * 70)


with open(
    JUDGE_FILE,
    "w",
    encoding="utf-8",
) as handle:

    for _, row in response_df.iterrows():

        judge_record = {
            "customer_message":
                row["customer_message"],

            "predicted_intent":
                row["predicted_intent"],

            "true_intent":
                row["true_intent"],

            "true_escalation":
                row["true_escalation"],

            "draft_response":
                row["draft_response"],

            "retrieval_top1_similarity":
                row["top1_similarity"],

            "retrieval_avg_top5_similarity":
                row["avg_top5_similarity"],

            # Judge fields to be filled later.
            "judge_grounded":
                None,

            "judge_helpful":
                None,

            "judge_correct":
                None,

            "judge_safe":
                None,

            "judge_reason":
                None,
        }

        handle.write(
            json.dumps(
                judge_record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    f"Rows prepared: {len(response_df)}"
)

print(
    f"Responses available: "
    f"{response_df['response_exists'].sum()}"
)

print(
    f"Internal-detail flags: "
    f"{response_df['contains_internal_details'].sum()}"
)

print(
    f"Unsupported-promise flags: "
    f"{response_df['contains_unsupported_promise'].sum()}"
)

print(
    f"Overly-long responses: "
    f"{response_df['too_long'].sum()}"
)

print(
    f"Local quality passes: "
    f"{response_df['local_quality_pass'].sum()}"
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("RESPONSE EVALUATION PREPARATION COMPLETE")
print("=" * 70)

print(
    "\nEvaluation file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nLLM judge input:"
)

print(
    JUDGE_FILE
)