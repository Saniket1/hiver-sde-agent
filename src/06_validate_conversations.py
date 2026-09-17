# ============================================================
# Hiver SDE Intern Assignment
# Step 9 - Validate AmazonHelp Conversation Dataset
# ============================================================
#
# PURPOSE:
# We have created the processed AmazonHelp conversation dataset.
#
# Before defining intents, we need to verify:
#
# 1. File exists
# 2. Required columns exist
# 3. No missing customer messages
# 4. No missing agent responses
# 5. No duplicate conversation IDs
# 6. Customer-message length distribution
# 7. Number of unique customers
# 8. Number of unique responses
# 9. Create a reproducible sample for intent discovery
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Automatically locate the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Processed AmazonHelp conversation dataset.
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)


# Intent-discovery sample output.
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_discovery_sample.csv"
)


# Make sure the processed-data folder exists.
SAMPLE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nConversation dataset not found:\n"
        f"{INPUT_PATH}\n\n"
        "Run 05_build_conversations.py first."
    )


# ============================================================
# 4. LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("HIVER SDE - CONVERSATION DATA VALIDATION")
print("=" * 70)

print(
    f"\nLoading:\n{INPUT_PATH}"
)


df = pd.read_csv(
    INPUT_PATH
)


# ============================================================
# 5. BASIC DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    "\nColumns:"
)

for column in df.columns:

    print(
        f"- {column}"
    )


# ============================================================
# 6. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "conversation_id",
    "customer_tweet_id",
    "customer_id",
    "customer_message",
    "agent_tweet_id",
    "agent_response",
    "timestamp",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


print(
    "\nRequired-column check: PASSED"
)


# ============================================================
# 7. MISSING VALUE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df[
    required_columns
].isna().sum()


print(
    missing_values
)


# ============================================================
# 8. DUPLICATE CHECKS
# ============================================================

print("\n")
print("=" * 70)
print("DUPLICATE CHECKS")
print("=" * 70)


duplicate_conversations = df[
    "conversation_id"
].duplicated().sum()


duplicate_customer_messages = df[
    "customer_message"
].duplicated().sum()


duplicate_pairs = df[
    [
        "customer_message",
        "agent_response",
    ]
].duplicated().sum()


print(
    f"Duplicate conversation IDs: "
    f"{duplicate_conversations:,}"
)

print(
    f"Duplicate customer messages: "
    f"{duplicate_customer_messages:,}"
)

print(
    f"Duplicate customer-response pairs: "
    f"{duplicate_pairs:,}"
)


# ============================================================
# 9. CUSTOMER / RESPONSE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CUSTOMER / RESPONSE STATISTICS")
print("=" * 70)


print(
    f"Unique customers: "
    f"{df['customer_id'].nunique():,}"
)


print(
    f"Unique customer messages: "
    f"{df['customer_message'].nunique():,}"
)


print(
    f"Unique agent responses: "
    f"{df['agent_response'].nunique():,}"
)


# ============================================================
# 10. MESSAGE LENGTH ANALYSIS
# ============================================================

# Convert message length into number of characters.
df["customer_message_length"] = (
    df["customer_message"]
    .astype(str)
    .str.len()
)


print("\n")
print("=" * 70)
print("CUSTOMER MESSAGE LENGTH")
print("=" * 70)


print(
    df["customer_message_length"]
    .describe()
)


# ============================================================
# 11. VERY SHORT MESSAGE CHECK
# ============================================================

# Extremely short messages may provide little information
# for intent discovery.
#
# We are NOT deleting them automatically here.
# We only measure them first.
very_short = df[
    df["customer_message_length"] < 10
]


print(
    f"\nMessages shorter than 10 characters: "
    f"{len(very_short):,}"
)


# ============================================================
# 12. TIMESTAMP CHECK
# ============================================================

print("\n")
print("=" * 70)
print("TIMESTAMP CHECK")
print("=" * 70)


timestamp_series = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


print(
    f"Invalid timestamps: "
    f"{timestamp_series.isna().sum():,}"
)


print(
    f"Date range: "
    f"{timestamp_series.min()} "
    f"to "
    f"{timestamp_series.max()}"
)


# ============================================================
# 13. CREATE INTENT DISCOVERY SAMPLE
# ============================================================
#
# The guidance recommends examining a sample of customer
# messages before defining the intent taxonomy.
#
# We use a fixed random_state so that the sample is
# reproducible.
# ============================================================


SAMPLE_SIZE = min(
    500,
    len(df)
)


intent_sample = (
    df[
        [
            "conversation_id",
            "customer_message",
            "agent_response",
            "timestamp",
        ]
    ]
    .sample(
        n=SAMPLE_SIZE,
        random_state=42
    )
    .reset_index(
        drop=True
    )
)


# Add an easy-to-reference sample ID.
intent_sample.insert(
    0,
    "sample_id",
    range(
        1,
        len(intent_sample) + 1
    )
)


# ============================================================
# 14. SAVE INTENT DISCOVERY SAMPLE
# ============================================================

intent_sample.to_csv(
    SAMPLE_PATH,
    index=False
)


# ============================================================
# 15. DISPLAY SAMPLE
# ============================================================

print("\n")
print("=" * 70)
print("INTENT DISCOVERY SAMPLE")
print("=" * 70)


print(
    f"\nSample size: "
    f"{len(intent_sample):,}"
)


print(
    intent_sample[
        [
            "sample_id",
            "customer_message",
            "agent_response",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE MESSAGE LENGTH SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)


print(
    f"\nIntent-discovery sample saved to:"
)

print(
    SAMPLE_PATH
)


print(
    f"\nOutput exists: "
    f"{SAMPLE_PATH.exists()}"
)