"""
Report Metrics Generator
------------------------

Purpose:
    Collect the measured results from all completed evaluation
    stages into one report-ready package.

Sources:
    - Final golden set
    - Intent baseline comparison
    - Escalation evaluation
    - Retrieval evaluation
    - Intent failure analysis
    - Human agreement check

Outputs:
    1. results/report_metrics_summary.csv
    2. results/report_metrics_summary.json
    3. results/report_failure_modes.csv
    4. results/report_notes.txt

Important:
    This script only summarizes measurements already produced.
    It does NOT invent or estimate missing metrics.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# Find the main project directory automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILES
# ============================================================

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "results"
    / "intent_baseline_comparison.csv"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_summary.csv"
)

FAILURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "intent_failure_analysis.csv"
)

AGREEMENT_FILE = (
    PROJECT_ROOT
    / "results"
    / "human_agreement_check.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_metrics_summary.csv"
)

SUMMARY_JSON = (
    PROJECT_ROOT
    / "results"
    / "report_metrics_summary.json"
)

FAILURES_CSV = (
    PROJECT_ROOT
    / "results"
    / "report_failure_modes.csv"
)

NOTES_FILE = (
    PROJECT_ROOT
    / "results"
    / "report_notes.txt"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = [
    GOLDEN_FILE,
    BASELINE_FILE,
    EVALUATION_FILE,
    FAILURE_FILE,
    AGREEMENT_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required report input not found:\n{file_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GENERATING REPORT METRICS")
print("=" * 70)


golden_df = pd.read_csv(
    GOLDEN_FILE
)

baseline_df = pd.read_csv(
    BASELINE_FILE
)

evaluation_df = pd.read_csv(
    EVALUATION_FILE
)

failure_df = pd.read_csv(
    FAILURE_FILE
)

agreement_df = pd.read_csv(
    AGREEMENT_FILE
)


# ============================================================
# GOLDEN SET METRICS
# ============================================================

golden_rows = len(
    golden_df
)

golden_intents = (
    golden_df["true_intent"]
    .nunique()
)

golden_missing_intents = (
    golden_df["true_intent"]
    .isna()
    .sum()
)

golden_missing_escalation = (
    golden_df["true_escalation"]
    .isna()
    .sum()
)

golden_duplicates = (
    golden_df["customer_message"]
    .duplicated()
    .sum()
)


# ============================================================
# INTENT BASELINE METRICS
# ============================================================

def get_metric(
    dataframe,
    model_name,
    metric_name,
):
    """
    Safely retrieve one model metric from the comparison table.
    """

    match = dataframe[
        dataframe["model"] == model_name
    ]

    if match.empty:
        return None

    value = match.iloc[0][
        metric_name
    ]

    return float(value)


majority_accuracy = get_metric(
    baseline_df,
    "Majority baseline",
    "accuracy",
)

majority_macro_f1 = get_metric(
    baseline_df,
    "Majority baseline",
    "macro_f1",
)

tfidf_accuracy = get_metric(
    baseline_df,
    "TF-IDF + Logistic Regression",
    "accuracy",
)

tfidf_macro_f1 = get_metric(
    baseline_df,
    "TF-IDF + Logistic Regression",
    "macro_f1",
)

embedding_accuracy = get_metric(
    baseline_df,
    "Sentence Embeddings + Nearest Centroid",
    "accuracy",
)

embedding_macro_f1 = get_metric(
    baseline_df,
    "Sentence Embeddings + Nearest Centroid",
    "macro_f1",
)


# ============================================================
# ESCALATION METRICS
# ============================================================

def get_evaluation_metric(
    dataframe,
    component,
    metric,
):
    """
    Retrieve a metric from evaluation_summary.csv.
    """

    match = dataframe[
        (dataframe["component"] == component)
        &
        (dataframe["metric"] == metric)
    ]

    if match.empty:

        return None

    return float(
        match.iloc[0]["value"]
    )


escalation_accuracy = get_evaluation_metric(
    evaluation_df,
    "Rule-based escalation",
    "accuracy",
)

human_precision = get_evaluation_metric(
    evaluation_df,
    "Rule-based escalation",
    "human_precision",
)

human_recall = get_evaluation_metric(
    evaluation_df,
    "Rule-based escalation",
    "human_recall",
)

human_f1 = get_evaluation_metric(
    evaluation_df,
    "Rule-based escalation",
    "human_f1",
)


# ============================================================
# RETRIEVAL METRICS
# ============================================================

avg_top1_similarity = get_evaluation_metric(
    evaluation_df,
    "Historical retrieval",
    "avg_top1_similarity",
)

avg_top5_similarity = get_evaluation_metric(
    evaluation_df,
    "Historical retrieval",
    "avg_top5_similarity",
)


# ============================================================
# HUMAN AGREEMENT
# ============================================================

# The agreement script already calculated these fields.
intent_agreement = (
    agreement_df[
        "intent_agreement"
    ]
    .mean()
)

escalation_agreement = (
    agreement_df[
        "escalation_agreement"
    ]
    .mean()
)

overall_agreement = (
    agreement_df[
        "overall_agreement"
    ]
    .mean()
)


# ============================================================
# FAILURE METRICS
# ============================================================

failure_count = len(
    failure_df
)

failure_rate = (
    failure_count
    / 48
)


# ============================================================
# MODEL IMPROVEMENT CALCULATIONS
# ============================================================

# Improvement of embedding classifier over TF-IDF.
embedding_vs_tfidf_accuracy_gain = (
    embedding_accuracy
    - tfidf_accuracy
)

embedding_vs_tfidf_f1_gain = (
    embedding_macro_f1
    - tfidf_macro_f1
)


# Improvement over majority baseline.
embedding_vs_majority_accuracy_gain = (
    embedding_accuracy
    - majority_accuracy
)

embedding_vs_majority_f1_gain = (
    embedding_macro_f1
    - majority_macro_f1
)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "section": "Golden set",
        "metric": "Human-verified examples",
        "value": golden_rows,
    },

    {
        "section": "Golden set",
        "metric": "Intents covered",
        "value": golden_intents,
    },

    {
        "section": "Golden set",
        "metric": "Missing intent labels",
        "value": golden_missing_intents,
    },

    {
        "section": "Golden set",
        "metric": "Missing escalation labels",
        "value": golden_missing_escalation,
    },

    {
        "section": "Golden set",
        "metric": "Duplicate messages",
        "value": golden_duplicates,
    },

    {
        "section": "Intent",
        "metric": "Majority baseline accuracy",
        "value": majority_accuracy,
    },

    {
        "section": "Intent",
        "metric": "Majority baseline macro F1",
        "value": majority_macro_f1,
    },

    {
        "section": "Intent",
        "metric": "TF-IDF accuracy",
        "value": tfidf_accuracy,
    },

    {
        "section": "Intent",
        "metric": "TF-IDF macro F1",
        "value": tfidf_macro_f1,
    },

    {
        "section": "Intent",
        "metric": "Embedding accuracy",
        "value": embedding_accuracy,
    },

    {
        "section": "Intent",
        "metric": "Embedding macro F1",
        "value": embedding_macro_f1,
    },

    {
        "section": "Intent",
        "metric": "Embedding accuracy gain vs TF-IDF",
        "value": embedding_vs_tfidf_accuracy_gain,
    },

    {
        "section": "Intent",
        "metric": "Embedding macro F1 gain vs TF-IDF",
        "value": embedding_vs_tfidf_f1_gain,
    },

    {
        "section": "Intent",
        "metric": "Embedding accuracy gain vs majority",
        "value": embedding_vs_majority_accuracy_gain,
    },

    {
        "section": "Intent",
        "metric": "Embedding macro F1 gain vs majority",
        "value": embedding_vs_majority_f1_gain,
    },

    {
        "section": "Escalation",
        "metric": "Rule accuracy",
        "value": escalation_accuracy,
    },

    {
        "section": "Escalation",
        "metric": "HUMAN precision",
        "value": human_precision,
    },

    {
        "section": "Escalation",
        "metric": "HUMAN recall",
        "value": human_recall,
    },

    {
        "section": "Escalation",
        "metric": "HUMAN F1",
        "value": human_f1,
    },

    {
        "section": "Retrieval",
        "metric": "Average top-1 similarity",
        "value": avg_top1_similarity,
    },

    {
        "section": "Retrieval",
        "metric": "Average top-5 similarity",
        "value": avg_top5_similarity,
    },

    {
        "section": "Failure analysis",
        "metric": "Intent errors on 48-example holdout",
        "value": failure_count,
    },

    {
        "section": "Failure analysis",
        "metric": "Intent error rate",
        "value": failure_rate,
    },

    {
        "section": "Human consistency",
        "metric": "Intent agreement",
        "value": intent_agreement,
    },

    {
        "section": "Human consistency",
        "metric": "Escalation agreement",
        "value": escalation_agreement,
    },

    {
        "section": "Human consistency",
        "metric": "Overall agreement",
        "value": overall_agreement,
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================
# CREATE JSON SUMMARY
# ============================================================

json_summary = {
    "golden_set": {
        "human_verified_examples": golden_rows,
        "intents_covered": golden_intents,
        "missing_intent_labels": int(
            golden_missing_intents
        ),
        "missing_escalation_labels": int(
            golden_missing_escalation
        ),
        "duplicate_messages": int(
            golden_duplicates
        ),
    },

    "intent": {
        "majority_accuracy": majority_accuracy,
        "majority_macro_f1": majority_macro_f1,
        "tfidf_accuracy": tfidf_accuracy,
        "tfidf_macro_f1": tfidf_macro_f1,
        "embedding_accuracy": embedding_accuracy,
        "embedding_macro_f1": embedding_macro_f1,
        "embedding_vs_tfidf_accuracy_gain":
            embedding_vs_tfidf_accuracy_gain,
        "embedding_vs_tfidf_f1_gain":
            embedding_vs_tfidf_f1_gain,
        "embedding_vs_majority_accuracy_gain":
            embedding_vs_majority_accuracy_gain,
        "embedding_vs_majority_f1_gain":
            embedding_vs_majority_f1_gain,
    },

    "escalation": {
        "accuracy": escalation_accuracy,
        "human_precision": human_precision,
        "human_recall": human_recall,
        "human_f1": human_f1,
    },

    "retrieval": {
        "average_top1_similarity":
            avg_top1_similarity,
        "average_top5_similarity":
            avg_top5_similarity,
    },

    "failure_analysis": {
        "errors_on_48_example_holdout":
            failure_count,
        "error_rate":
            failure_rate,
    },

    "human_consistency": {
        "intent_agreement":
            intent_agreement,
        "escalation_agreement":
            escalation_agreement,
        "overall_agreement":
            overall_agreement,
        "type":
            "intra-annotator consistency",
    },
}


# Write JSON summary.
with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        json_summary,
        file,
        indent=2,
    )


# ============================================================
# CREATE REPORT FAILURE TABLE
# ============================================================

# Count the most frequent confusion pairs.
failure_modes = (
    failure_df
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

# Add a readable failure-mode label.
failure_modes[
    "failure_mode"
] = (
    failure_modes[
        "true_intent"
    ]
    + " -> "
    + failure_modes[
        "predicted_intent"
    ]
)

# Reorder columns.
failure_modes = failure_modes[
    [
        "failure_mode",
        "true_intent",
        "predicted_intent",
        "count",
    ]
]

# Save top failure modes.
failure_modes.to_csv(
    FAILURES_CSV,
    index=False,
)


# ============================================================
# REPORT NOTES
# ============================================================

notes = f"""
REPORT METRICS NOTES
====================

Golden set
----------
Human-verified examples: {golden_rows}
Intents covered: {golden_intents}
Duplicate customer messages: {golden_duplicates}
Missing intent labels: {golden_missing_intents}
Missing escalation labels: {golden_missing_escalation}

Intent evaluation
-----------------
Majority accuracy: {majority_accuracy:.4f}
Majority macro F1: {majority_macro_f1:.4f}

TF-IDF accuracy: {tfidf_accuracy:.4f}
TF-IDF macro F1: {tfidf_macro_f1:.4f}

Embedding accuracy: {embedding_accuracy:.4f}
Embedding macro F1: {embedding_macro_f1:.4f}

Embedding accuracy gain vs TF-IDF:
{embedding_vs_tfidf_accuracy_gain:.4f}

Embedding macro F1 gain vs TF-IDF:
{embedding_vs_tfidf_f1_gain:.4f}

Embedding accuracy gain vs majority:
{embedding_vs_majority_accuracy_gain:.4f}

Escalation evaluation
---------------------
Accuracy: {escalation_accuracy:.4f}
HUMAN precision: {human_precision:.4f}
HUMAN recall: {human_recall:.4f}
HUMAN F1: {human_f1:.4f}

Retrieval
---------
Average top-1 similarity: {avg_top1_similarity:.4f}
Average top-5 similarity: {avg_top5_similarity:.4f}

Failure analysis
----------------
Errors on 48-example intent holdout: {failure_count}
Intent error rate: {failure_rate:.2%}

Human consistency
-----------------
Intent agreement: {intent_agreement:.2%}
Escalation agreement: {escalation_agreement:.2%}
Overall agreement: {overall_agreement:.2%}

The agreement test is intra-annotator consistency, not independent
multi-annotator agreement.

Response-quality evaluation
---------------------------
No final LLM-as-judge response-quality score is included yet because
Gemini API quota was exhausted during response-generation testing.
Do not present a response-quality metric until actual generated
responses have been judged.

Evaluation-size limitation
--------------------------
The intent holdout contains 48 examples. Several intents have only
one or two test examples, so per-intent metrics are unstable and
should not be interpreted as population-level performance.

Escalation-label limitation
---------------------------
Second-pass escalation agreement was low relative to intent agreement.
Escalation conclusions should therefore be treated cautiously.

"""
# Save report notes.
with open(
    NOTES_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        notes.strip()
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("REPORT METRICS GENERATED")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSummary CSV:")
print(
    SUMMARY_CSV
)

print("\nSummary JSON:")
print(
    SUMMARY_JSON
)

print("\nFailure modes:")
print(
    FAILURES_CSV
)

print("\nReport notes:")
print(
    NOTES_FILE
)