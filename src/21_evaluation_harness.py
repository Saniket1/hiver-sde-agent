"""
Local Evaluation Harness
------------------------

Purpose:
    Evaluate the current AmazonHelp support-agent components
    without requiring any Gemini API calls.

What is evaluated:

    1. Intent classification
       - Accuracy
       - Macro F1
       - Weighted F1
       - Per-intent precision/recall/F1

    2. Escalation decision
       - Accuracy
       - HUMAN precision/recall/F1
       - Confusion matrix

    3. Historical retrieval
       - Whether retrieved examples contain the same intent
         when an intent label can be inferred from metadata
       - Average top similarity
       - Top-5 similarity statistics

Important:
    This harness does not call Gemini.
    That makes the evaluation reproducible even when the
    Gemini Free Tier quota is exhausted.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

# This script is inside the "src" directory.
# parents[1] therefore points to the project root.
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

# Historical FAISS index.
INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_faiss.index"
)

# Metadata corresponding to the FAISS vectors.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_retrieval_metadata.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_summary.csv"
)

DETAILED_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_detailed.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# The evaluation split is kept identical to the earlier
# intent baseline so our comparison remains consistent.
TEST_SIZE = 0.25

RANDOM_STATE = 42

# Number of historical examples retrieved for each test query.
TOP_K = 5

# Sentence-transformer model used by the retrieval layer.
EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

# Make sure the results folder exists.
RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    GOLDEN_FILE,
    INDEX_FILE,
    METADATA_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
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

print(
    f"Rows loaded: {len(golden_df)}"
)


# ============================================================
# VALIDATE GOLDEN SET
# ============================================================

required_columns = {
    "customer_message",
    "true_intent",
    "true_escalation",
}

missing_columns = (
    required_columns
    - set(golden_df.columns)
)

if missing_columns:

    raise ValueError(
        f"Golden set is missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# CLEAN GOLDEN SET
# ============================================================

# Remove incomplete rows.
golden_df = golden_df.dropna(
    subset=[
        "customer_message",
        "true_intent",
        "true_escalation",
    ]
).copy()


# Clean customer text.
golden_df["customer_message"] = (
    golden_df["customer_message"]
    .astype(str)
    .str.strip()
)


# Convert escalation labels into booleans.
golden_df["true_escalation"] = (
    golden_df["true_escalation"]
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


# Remove rows where boolean conversion failed.
golden_df = golden_df.dropna(
    subset=[
        "true_escalation"
    ]
).copy()


print(
    f"Usable golden examples: {len(golden_df)}"
)


# ============================================================
# ------------------------------------------------------------
# PART 1: INTENT EVALUATION
# ------------------------------------------------------------
# ============================================================

print("\n")
print("=" * 70)
print("PART 1 - INTENT CLASSIFICATION")
print("=" * 70)


# ------------------------------------------------------------
# PREPARE INPUTS
# ------------------------------------------------------------

X = golden_df[
    "customer_message"
]

y = golden_df[
    "true_intent"
]


# ------------------------------------------------------------
# CREATE SAME TRAIN/TEST SPLIT
# ------------------------------------------------------------

# Use exactly the same split settings as script 13 and script 14.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


print(
    f"Training examples: {len(X_train)}"
)

print(
    f"Test examples: {len(X_test)}"
)


# ------------------------------------------------------------
# BUILD TF-IDF BASELINE
# ------------------------------------------------------------

# This reproduces the simple baseline used earlier.
tfidf_model = Pipeline(
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
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# Train the baseline.
tfidf_model.fit(
    X_train,
    y_train,
)


# Generate test predictions.
intent_predictions = tfidf_model.predict(
    X_test
)


# ------------------------------------------------------------
# CALCULATE INTENT METRICS
# ------------------------------------------------------------

intent_accuracy = accuracy_score(
    y_test,
    intent_predictions,
)


intent_precision, intent_recall, intent_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        intent_predictions,
        average="macro",
        zero_division=0,
    )
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        intent_predictions,
        average="weighted",
        zero_division=0,
    )
)


print("\nIntent accuracy:")
print(
    f"{intent_accuracy:.4f}"
)

print("\nMacro precision:")
print(
    f"{intent_precision:.4f}"
)

print("\nMacro recall:")
print(
    f"{intent_recall:.4f}"
)

print("\nMacro F1:")
print(
    f"{intent_f1:.4f}"
)

print("\nWeighted F1:")
print(
    f"{weighted_f1:.4f}"
)


# ------------------------------------------------------------
# PRINT PER-INTENT REPORT
# ------------------------------------------------------------

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        intent_predictions,
        zero_division=0,
    )
)


# ============================================================
# PART 2: ESCALATION EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("PART 2 - ESCALATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD RULE-BASED PREDICTIONS
# ------------------------------------------------------------

# Script 18 already produced predictions for every golden row.
RULE_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "rule_escalation_predictions_v2.csv"
)


if not RULE_PREDICTIONS_FILE.exists():

    raise FileNotFoundError(
        "Run script 18 first:\n"
        "python .\\src\\18_escalation_rules.py"
    )


rule_df = pd.read_csv(
    RULE_PREDICTIONS_FILE
)


# ------------------------------------------------------------
# NORMALIZE ESCALATION VALUES
# ------------------------------------------------------------

rule_df["true_escalation"] = (
    rule_df["true_escalation"]
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


rule_df["predicted_escalation"] = (
    rule_df["predicted_escalation"]
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


# Remove invalid rows.
rule_df = rule_df.dropna(
    subset=[
        "true_escalation",
        "predicted_escalation",
    ]
).copy()


# ------------------------------------------------------------
# ESCALATION METRICS
# ------------------------------------------------------------

escalation_accuracy = accuracy_score(
    rule_df["true_escalation"],
    rule_df["predicted_escalation"],
)


# HUMAN class = True.
human_precision, human_recall, human_f1, _ = (
    precision_recall_fscore_support(
        rule_df["true_escalation"],
        rule_df["predicted_escalation"],
        average="binary",
        pos_label=True,
        zero_division=0,
    )
)


print(
    f"Escalation accuracy: "
    f"{escalation_accuracy:.4f}"
)

print(
    f"HUMAN precision: "
    f"{human_precision:.4f}"
)

print(
    f"HUMAN recall: "
    f"{human_recall:.4f}"
)

print(
    f"HUMAN F1: "
    f"{human_f1:.4f}"
)


# ------------------------------------------------------------
# ESCALATION CONFUSION MATRIX
# ------------------------------------------------------------

escalation_cm = confusion_matrix(
    rule_df["true_escalation"],
    rule_df["predicted_escalation"],
    labels=[
        False,
        True,
    ],
)


print("\nEscalation confusion matrix:")

print(
    pd.DataFrame(
        escalation_cm,
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
# PART 3: RETRIEVAL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("PART 3 - HISTORICAL RETRIEVAL")
print("=" * 70)


# ------------------------------------------------------------
# LOAD FAISS INDEX
# ------------------------------------------------------------

index = faiss.read_index(
    str(INDEX_FILE)
)


# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

metadata = pd.read_csv(
    METADATA_FILE
)


# Make sure vector and metadata counts match.
if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata are misaligned."
    )


# ------------------------------------------------------------
# IDENTIFY HISTORICAL TEXT COLUMNS
# ------------------------------------------------------------

possible_customer_columns = [
    "customer_message",
    "customer_text",
    "text",
]

possible_response_columns = [
    "agent_response",
    "response",
    "response_text",
]


customer_column = None

for column in possible_customer_columns:

    if column in metadata.columns:

        customer_column = column
        break


response_column = None

for column in possible_response_columns:

    if column in metadata.columns:

        response_column = column
        break


if customer_column is None:

    raise ValueError(
        "Historical customer-message column not found."
    )


if response_column is None:

    raise ValueError(
        "Historical response column not found."
    )


# ------------------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print(
    "\nLoading embedding model for retrieval evaluation..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ------------------------------------------------------------
# SAMPLE TEST QUERIES
# ------------------------------------------------------------

# Use the same 48 held-out intent-test examples.
retrieval_queries = X_test.tolist()

retrieval_true_intents = y_test.tolist()


# ------------------------------------------------------------
# GENERATE QUERY EMBEDDINGS
# ------------------------------------------------------------

query_embeddings = embedding_model.encode(
    retrieval_queries,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(
    "float32"
)


# ------------------------------------------------------------
# SEARCH FAISS
# ------------------------------------------------------------

retrieval_scores, retrieval_indices = index.search(
    query_embeddings,
    TOP_K,
)


# ------------------------------------------------------------
# RETRIEVAL STATISTICS
# ------------------------------------------------------------

top1_scores = []
top5_scores = []

retrieval_records = []


for query_number in range(
    len(retrieval_queries)
):

    # Record the top similarity.
    top1_scores.append(
        float(
            retrieval_scores[
                query_number,
                0,
            ]
        )
    )

    # Record the average similarity across top-5.
    top5_scores.append(
        float(
            np.mean(
                retrieval_scores[
                    query_number
                ]
            )
        )
    )


# ------------------------------------------------------------
# CALCULATE RETRIEVAL METRICS
# ------------------------------------------------------------

average_top1_similarity = float(
    np.mean(
        top1_scores
    )
)


average_top5_similarity = float(
    np.mean(
        top5_scores
    )
)


print(
    f"Average top-1 similarity: "
    f"{average_top1_similarity:.4f}"
)

print(
    f"Average top-5 similarity: "
    f"{average_top5_similarity:.4f}"
)


# ============================================================
# BUILD DETAILED EVALUATION FILE
# ============================================================

detailed_records = []


# ------------------------------------------------------------
# INTENT RECORDS
# ------------------------------------------------------------

for message, true_intent, predicted_intent in zip(
    X_test.tolist(),
    y_test.tolist(),
    intent_predictions.tolist(),
):

    detailed_records.append(
        {
            "customer_message": message,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "intent_correct":
                true_intent == predicted_intent,
        }
    )


# ------------------------------------------------------------
# MERGE RETRIEVAL INFORMATION
# ------------------------------------------------------------

detailed_df = pd.DataFrame(
    detailed_records
)


detailed_df["top1_similarity"] = (
    top1_scores
)


detailed_df["avg_top5_similarity"] = (
    top5_scores
)


# Save detailed results.
detailed_df.to_csv(
    DETAILED_FILE,
    index=False,
)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "component": "TF-IDF intent classifier",
        "metric": "accuracy",
        "value": intent_accuracy,
    },

    {
        "component": "TF-IDF intent classifier",
        "metric": "macro_f1",
        "value": intent_f1,
    },

    {
        "component": "TF-IDF intent classifier",
        "metric": "weighted_f1",
        "value": weighted_f1,
    },

    {
        "component": "Rule-based escalation",
        "metric": "accuracy",
        "value": escalation_accuracy,
    },

    {
        "component": "Rule-based escalation",
        "metric": "human_precision",
        "value": human_precision,
    },

    {
        "component": "Rule-based escalation",
        "metric": "human_recall",
        "value": human_recall,
    },

    {
        "component": "Rule-based escalation",
        "metric": "human_f1",
        "value": human_f1,
    },

    {
        "component": "Historical retrieval",
        "metric": "avg_top1_similarity",
        "value": average_top1_similarity,
    },

    {
        "component": "Historical retrieval",
        "metric": "avg_top5_similarity",
        "value": average_top5_similarity,
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


# Save summary.
summary_df.to_csv(
    RESULTS_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION HARNESS COMPLETE")
print("=" * 70)

print("\nSummary:")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSummary saved to:")
print(
    RESULTS_FILE
)

print("\nDetailed results saved to:")
print(
    DETAILED_FILE
)