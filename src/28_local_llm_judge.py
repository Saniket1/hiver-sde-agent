
"""
Hiver SDE Intern Assignment
Fast Local LLM-as-Judge

This script:
1. Loads the already-generated 10 evaluation examples.
2. Uses FLAN-T5-small as a lightweight local LLM judge.
3. Scores each reply on:
   - relevance
   - groundedness
   - safety
   - conciseness
   - overall quality
4. Saves the judge scores.
5. Creates a human-review CSV for judge-vs-human agreement.

Important:
The draft replies are already present in the previous CSV, so this
script does NOT generate them again.
"""

from pathlib import Path
import json
import re

import pandas as pd

# Directly use the small sequence-to-sequence model instead of the
# text-generation pipeline. This avoids the previous pipeline issue.
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Existing file produced by the previous evaluation attempt.
INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "local_llm_judge_results.csv"
)

# File containing final LLM-judge scores.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "local_llm_judge_results.csv"
)

# File to be manually reviewed by the human annotator.
HUMAN_REVIEW_PATH = (
    PROJECT_ROOT
    / "results"
    / "human_response_quality_review.csv"
)


# ============================================================
# 2. LOAD EXISTING EVALUATION DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print(f"Evaluation rows loaded: {len(df)}")


# ============================================================
# 3. LOAD SMALL LOCAL MODEL
# ============================================================

# FLAN-T5-small is substantially smaller than Qwen2.5-0.5B
# and is designed for instruction-style text-to-text tasks.
MODEL_NAME = "google/flan-t5-small"

print("\nLoading lightweight local judge model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 4. FUNCTION TO BUILD THE JUDGE PROMPT
# ============================================================

def build_prompt(row):
    """
    Build a compact evaluation prompt for one support response.

    The prompt asks for integer scores from 1 to 5.
    """

    return f"""
Evaluate this customer-support reply.

Customer:
{row["customer_message"]}

Historical resolution:
{row["historical_resolution"]}

Draft reply:
{row["draft_reply"]}

Score from 1 to 5:

Relevance:
Does the reply address the customer's problem?

Groundedness:
Is the reply supported by the historical resolution?

Safety:
Does it avoid unsupported promises or invented actions?

Conciseness:
Is it clear and brief?

Overall:
Overall quality.

Return ONLY this format:

relevance=NUMBER
groundedness=NUMBER
safety=NUMBER
conciseness=NUMBER
overall=NUMBER
"""


# ============================================================
# 5. FUNCTION TO RUN ONE JUDGE REQUEST
# ============================================================

def run_judge(prompt):
    """
    Run FLAN-T5-small and return its generated text.

    A short output limit keeps CPU inference manageable.
    """

    # Convert the prompt into model tokens.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Generate a short deterministic answer.
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
    )

    # Convert generated tokens back to plain text.
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 6. PARSE SCORE TEXT
# ============================================================

def extract_score(text, field):
    """
    Extract a 1-5 score from output such as:
        relevance=4
    """

    pattern = rf"{field}\s*=\s*([1-5])"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 7. RUN THE LLM JUDGE
# ============================================================

results = []

for _, row in df.iterrows():

    print(
        f"Judging example {row['sample_id']}..."
    )

    # Build the evaluation prompt.
    prompt = build_prompt(row)

    # Ask the local LLM to judge the reply.
    raw_output = run_judge(prompt)

    # Extract individual scores.
    relevance = extract_score(
        raw_output,
        "relevance",
    )

    groundedness = extract_score(
        raw_output,
        "groundedness",
    )

    safety = extract_score(
        raw_output,
        "safety",
    )

    conciseness = extract_score(
        raw_output,
        "conciseness",
    )

    overall = extract_score(
        raw_output,
        "overall",
    )

    # Store the judge result.
    result = row.to_dict()

    result["judge_raw_output"] = raw_output
    result["judge_relevance"] = relevance
    result["judge_groundedness"] = groundedness
    result["judge_safety"] = safety
    result["judge_conciseness"] = conciseness
    result["judge_overall"] = overall

    results.append(result)


# ============================================================
# 8. SAVE JUDGE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. CREATE HUMAN REVIEW FILE
# ============================================================

human_review = result_df[
    [
        "sample_id",
        "customer_message",
        "draft_reply",
        "historical_resolution",
        "judge_overall",
    ]
].copy()

# These columns are intentionally left blank for human scoring.
human_review["human_overall"] = ""
human_review["human_reason"] = ""

human_review.to_csv(
    HUMAN_REVIEW_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

valid_scores = result_df[
    result_df["judge_overall"].notna()
]

print("\n" + "=" * 60)
print("FAST LOCAL LLM-AS-JUDGE COMPLETE")
print("=" * 60)

print(
    f"Examples evaluated: {len(result_df)}"
)

print(
    f"Valid overall judge scores: "
    f"{len(valid_scores)}"
)

print(
    f"\nJudge results:\n{OUTPUT_PATH}"
)

print(
    f"\nHuman review file:\n{HUMAN_REVIEW_PATH}"
)

print(
    "\nThe next step is to fill human_overall "
    "for the 10 examples."
)