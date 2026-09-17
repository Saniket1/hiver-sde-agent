# ============================================================
# Hiver SDE Intern Assignment
# Step 12 - Prepare Golden-Set Candidates
# ============================================================
#
# PURPOSE:
# Prepare a candidate set of 200 AmazonHelp examples for the
# hand-labelled golden evaluation dataset.
#
# IMPORTANT:
# This script DOES NOT create the final ground-truth labels.
#
# It only selects candidate examples.
#
# Final:
#     true_intent
#     true_escalation
#
# must be manually verified by the student.
# ============================================================

from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Locate the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Clean AmazonHelp conversation dataset.
CONVERSATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)

# Frozen intent taxonomy.
INTENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intents.json"
)

# Candidate golden-set output.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_candidates.csv"
)

# Make sure the golden directory exists.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

# We want a final golden set of 200 examples.
TOTAL_GOLDEN_EXAMPLES = 200

# We aim for approximately 20 examples per intent.
EXAMPLES_PER_INTENT = 20

# Fixed seed makes the sampling reproducible.
RANDOM_STATE = 42


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

if not CONVERSATIONS_PATH.exists():
    raise FileNotFoundError(
        f"Conversation dataset not found:\n"
        f"{CONVERSATIONS_PATH}"
    )

if not INTENTS_PATH.exists():
    raise FileNotFoundError(
        f"Intent taxonomy not found:\n"
        f"{INTENTS_PATH}"
    )


# ============================================================
# 4. LOAD TAXONOMY
# ============================================================

with open(
    INTENTS_PATH,
    "r",
    encoding="utf-8"
) as file:

    taxonomy = json.load(file)


# Extract final intent names.
intent_names = [
    intent["name"]
    for intent in taxonomy["intents"]
]


print("=" * 70)
print("HIVER SDE - GOLDEN SET CANDIDATE PREPARATION")
print("=" * 70)

print(
    f"\nNumber of final intents: "
    f"{len(intent_names)}"
)

for intent in intent_names:
    print(
        f"- {intent}"
    )


# ============================================================
# 5. LOAD CONVERSATIONS
# ============================================================

print(
    "\nLoading AmazonHelp conversations..."
)

df = pd.read_csv(
    CONVERSATIONS_PATH
)


# Remove rows without usable customer messages.
df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["customer_message"].ne("")
].copy()


# Remove very short messages for the candidate pool.
#
# We are not saying these are invalid forever.
# We simply want the first golden-set candidate pool to contain
# messages with enough context to label reliably.
df = df[
    df["customer_message"].str.len() >= 10
].copy()


# Remove exact duplicate customer messages from the candidate
# pool so one repeated message does not dominate the sample.
df = df.drop_duplicates(
    subset=["customer_message"]
).copy()


print(
    f"\nUsable candidate messages: "
    f"{len(df):,}"
)


# ============================================================
# 6. CREATE REPRODUCIBLE RANDOM SAMPLE
# ============================================================
#
# At this stage we DON'T know the true intent of every row.
# Therefore we cannot genuinely stratify by intent yet.
#
# We create a broad candidate pool that will later be manually
# labelled according to the frozen taxonomy.
# ============================================================

candidate_size = min(
    1000,
    len(df)
)


candidates = df.sample(
    n=candidate_size,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


# Give each candidate a stable ID.
candidates.insert(
    0,
    "golden_candidate_id",
    range(
        1,
        len(candidates) + 1
    )
)


# ============================================================
# 7. SELECT INITIAL 200
# ============================================================
#
# We select 200 candidates from this reproducible pool.
# Later, during manual review, we can replace ambiguous or
# duplicate examples with other candidates from the remaining
# 800 rows.
# ============================================================

golden_candidates = candidates.head(
    TOTAL_GOLDEN_EXAMPLES
).copy()


# ============================================================
# 8. CREATE EMPTY LABEL COLUMNS
# ============================================================

# IMPORTANT:
# These columns are intentionally empty.
#
# They are the ground-truth labels that YOU will verify.
golden_candidates["true_intent"] = ""

golden_candidates["true_escalation"] = ""


# ============================================================
# 9. ADD LABELING NOTES COLUMN
# ============================================================

# This will help us document difficult/ambiguous examples
# during manual review.
golden_candidates["labeling_notes"] = ""


# ============================================================
# 10. SELECT FINAL COLUMNS
# ============================================================

golden_candidates = golden_candidates[
    [
        "golden_candidate_id",
        "conversation_id",
        "customer_message",
        "agent_response",
        "timestamp",
        "true_intent",
        "true_escalation",
        "labeling_notes",
    ]
]


# ============================================================
# 11. SAVE CANDIDATES
# ============================================================

golden_candidates.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("GOLDEN CANDIDATES CREATED")
print("=" * 70)

print(
    f"\nCandidate pool: "
    f"{candidate_size:,}"
)

print(
    f"Selected candidates: "
    f"{len(golden_candidates):,}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nOutput exists: "
    f"{OUTPUT_PATH.exists()}"
)

print(
    "\nIMPORTANT:"
)

print(
    "true_intent and true_escalation are intentionally empty."
)

print(
    "They must be manually verified before the golden set "
    "is considered ground truth."
)