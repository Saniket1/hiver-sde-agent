# ============================================================
# Hiver SDE Intern Assignment
# Step 8 - Build AmazonHelp Customer-Support Conversations
# ============================================================
#
# PURPOSE:
# Convert the raw Twitter dataset into clean customer-support
# examples for the selected brand: AmazonHelp.
#
# Each output row represents:
#
#     customer message
#             +
#     AmazonHelp response
#
# Output:
#
# data/processed/amazonhelp_conversations.csv
#
# The resulting dataset will later be used for:
#   1. Intent discovery
#   2. Golden-set creation
#   3. Historical retrieval / FAISS
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

# Identify the project root automatically.
#
# Current file:
# hiver-sde-agent/src/05_build_conversations.py
#
# parent       -> src
# parent.parent -> hiver-sde-agent
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Original Customer Support on Twitter dataset.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "twcs.csv"
)


# Final processed conversation dataset.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)


# Make sure the output directory exists.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

# This is the brand we selected after our brand-selection
# analysis.
BRAND_NAME = "AmazonHelp"


# The raw file is large, so process it in chunks.
CHUNK_SIZE = 100_000


# ============================================================
# 4. CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure twcs.csv is located inside data/raw/"
    )


print("=" * 70)
print("HIVER SDE - AMAZONHELP CONVERSATION BUILDER")
print("=" * 70)

print(f"\nSelected brand: {BRAND_NAME}")
print(f"Dataset: {DATA_PATH}")
print(f"Output: {OUTPUT_PATH}")


# ============================================================
# 5. PASS 1
# FIND ALL AMAZONHELP RESPONSES
# ============================================================
#
# We first identify every AmazonHelp outbound tweet that
# directly responds to another tweet.
#
# Example:
#
# Customer tweet ID = 12345
# AmazonHelp response has:
#
#     author_id = AmazonHelp
#     inbound = False
#     in_response_to_tweet_id = 12345
#
# We store the relationship for the second pass.
# ============================================================

brand_responses = []


print("\n")
print("=" * 70)
print("PASS 1 - FIND AMAZONHELP RESPONSES")
print("=" * 70)


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,

        # Only load columns required for this operation.
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id",
        ],

        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),

    start=1,
):

    print(
        f"Processing chunk {chunk_number}..."
    )


    # Keep only AmazonHelp messages.
    amazon_rows = chunk[
        chunk["author_id"]
        .astype(str)
        .eq(BRAND_NAME)
    ].copy()


    # AmazonHelp's support responses are outbound.
    amazon_rows = amazon_rows[
        amazon_rows["inbound"] == False
    ]


    # A useful response needs a parent tweet.
    amazon_rows = amazon_rows[
        amazon_rows["in_response_to_tweet_id"].notna()
    ]


    if amazon_rows.empty:
        continue


    # Convert parent tweet ID to numeric.
    amazon_rows["parent_tweet_id"] = pd.to_numeric(
        amazon_rows["in_response_to_tweet_id"],
        errors="coerce"
    )


    # Remove invalid parent IDs.
    amazon_rows = amazon_rows[
        amazon_rows["parent_tweet_id"].notna()
    ]


    if amazon_rows.empty:
        continue


    # Use integer IDs so they match the original tweet_id type.
    amazon_rows["parent_tweet_id"] = (
        amazon_rows["parent_tweet_id"]
        .astype("int64")
    )


    # Keep only what we need to join against customer tweets.
    brand_responses.append(
        amazon_rows[
            [
                "tweet_id",
                "created_at",
                "text",
                "parent_tweet_id",
            ]
        ]
    )


# Make sure we found support responses.
if not brand_responses:

    raise RuntimeError(
        "No AmazonHelp response tweets were found."
    )


# Combine all chunks.
brand_response_df = pd.concat(
    brand_responses,
    ignore_index=True
)


print(
    f"\nAmazonHelp responses found: "
    f"{len(brand_response_df):,}"
)


# ============================================================
# 6. GET CUSTOMER TWEET IDS
# ============================================================

# These are the customer tweet IDs that AmazonHelp responded to.
customer_tweet_ids = set(
    brand_response_df["parent_tweet_id"]
)


print(
    f"Customer tweets to retrieve: "
    f"{len(customer_tweet_ids):,}"
)


# ============================================================
# 7. PASS 2
# RETRIEVE CUSTOMER TWEETS
# ============================================================
#
# Now we scan the original dataset again and retrieve the
# customer tweets referenced by AmazonHelp responses.
# ============================================================

customer_rows = []


print("\n")
print("=" * 70)
print("PASS 2 - RETRIEVE CUSTOMER MESSAGES")
print("=" * 70)


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,

        # Only load columns needed to build the final pair.
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
        ],

        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),

    start=1,
):

    print(
        f"Processing chunk {chunk_number}..."
    )


    # Find tweets referenced by AmazonHelp responses.
    matching = chunk[
        chunk["tweet_id"].isin(
            customer_tweet_ids
        )
    ].copy()


    if matching.empty:
        continue


    # Keep actual customer/inbound messages.
    matching = matching[
        matching["inbound"] == True
    ]


    if matching.empty:
        continue


    customer_rows.append(
        matching
    )


# Make sure customer tweets were found.
if not customer_rows:

    raise RuntimeError(
        "No customer tweets were found."
    )


# Combine all customer rows.
customer_df = pd.concat(
    customer_rows,
    ignore_index=True
)


print(
    f"\nCustomer tweets retrieved: "
    f"{len(customer_df):,}"
)


# ============================================================
# 8. RENAME COLUMNS
# ============================================================

# Rename columns so it is obvious which fields belong to
# the customer and which belong to AmazonHelp.
customer_df = customer_df.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "author_id": "customer_id",
        "created_at": "customer_created_at",
        "text": "customer_message",
    }
)


brand_response_df = brand_response_df.rename(
    columns={
        "tweet_id": "agent_tweet_id",
        "created_at": "agent_created_at",
        "text": "agent_response",
    }
)


# ============================================================
# 9. JOIN CUSTOMER + AMAZONHELP RESPONSE
# ============================================================

# parent_tweet_id in the AmazonHelp response points to the
# customer's tweet.
conversation_df = brand_response_df.merge(
    customer_df,
    left_on="parent_tweet_id",
    right_on="customer_tweet_id",
    how="inner"
)


print(
    f"\nCustomer-response pairs created: "
    f"{len(conversation_df):,}"
)


# ============================================================
# 10. BASIC CLEANING
# ============================================================

# Remove rows with missing customer messages.
conversation_df = conversation_df[
    conversation_df["customer_message"].notna()
]


# Remove rows with missing AmazonHelp responses.
conversation_df = conversation_df[
    conversation_df["agent_response"].notna()
]


# Convert text fields to strings and strip whitespace.
conversation_df["customer_message"] = (
    conversation_df["customer_message"]
    .astype(str)
    .str.strip()
)

conversation_df["agent_response"] = (
    conversation_df["agent_response"]
    .astype(str)
    .str.strip()
)


# Remove completely empty customer messages.
conversation_df = conversation_df[
    conversation_df["customer_message"].ne("")
]


# Remove completely empty responses.
conversation_df = conversation_df[
    conversation_df["agent_response"].ne("")
]


# ============================================================
# 11. REMOVE EXACT DUPLICATES
# ============================================================
#
# Exact duplicate customer-response pairs are not useful for
# intent discovery or retrieval evaluation.
# ============================================================

conversation_df = conversation_df.drop_duplicates(
    subset=[
        "customer_message",
        "agent_response",
    ]
)


# ============================================================
# 12. CREATE CONVERSATION ID
# ============================================================
#
# For this first version, each directly linked
# customer-response interaction receives a stable ID.
#
# Later, we can reconstruct larger multi-turn threads where
# necessary.
# ============================================================

conversation_df["conversation_id"] = (
    "amazonhelp_"
    + conversation_df["customer_tweet_id"]
    .astype(str)
)


# ============================================================
# 13. CREATE FINAL TIMESTAMP
# ============================================================
#
# The customer message timestamp is useful for chronological
# analysis and report documentation.
# ============================================================

conversation_df["timestamp"] = (
    conversation_df["customer_created_at"]
)


# ============================================================
# 14. SELECT FINAL COLUMNS
# ============================================================

final_df = conversation_df[
    [
        "conversation_id",
        "customer_tweet_id",
        "customer_id",
        "customer_message",
        "agent_tweet_id",
        "agent_response",
        "timestamp",
    ]
].copy()


# ============================================================
# 15. SORT BY TIMESTAMP
# ============================================================

# Convert timestamp strings to datetime where possible.
final_df["timestamp"] = pd.to_datetime(
    final_df["timestamp"],
    errors="coerce"
)


# Sort oldest → newest.
final_df = final_df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# 16. FINAL QUALITY CHECKS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print(
    f"Rows: {len(final_df):,}"
)

print(
    f"Unique customers: "
    f"{final_df['customer_id'].nunique():,}"
)

print(
    f"Unique customer messages: "
    f"{final_df['customer_message'].nunique():,}"
)

print(
    f"Unique agent responses: "
    f"{final_df['agent_response'].nunique():,}"
)

print(
    f"Missing customer messages: "
    f"{final_df['customer_message'].isna().sum():,}"
)

print(
    f"Missing agent responses: "
    f"{final_df['agent_response'].isna().sum():,}"
)

# Count duplicate customer-response pairs.
duplicate_count = final_df.duplicated(
    subset=[
        "customer_message",
        "agent_response",
    ]
).sum()

print(
    f"Duplicate customer-response pairs: "
    f"{duplicate_count:,}"
)

# ============================================================
# 17. SAVE PROCESSED DATASET
# ============================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CONVERSATION BUILD COMPLETE")
print("=" * 70)

print(
    f"\nSelected brand: {BRAND_NAME}"
)

print(
    f"Output rows: {len(final_df):,}"
)

print(
    f"Saved to:\n{OUTPUT_PATH}"
)

print(
    f"\nOutput exists: {OUTPUT_PATH.exists()}"
)


# ============================================================
# 19. SHOW SAMPLE RECORDS
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE CUSTOMER-SUPPORT RECORDS")
print("=" * 70)

print(
    final_df[
        [
            "conversation_id",
            "customer_message",
            "agent_response",
            "timestamp",
        ]
    ]
    .head(10)
    .to_string(index=False)
)