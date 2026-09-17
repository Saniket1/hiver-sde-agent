# ============================================================
# Hiver SDE Intern Assignment
# Step 4 - Historical Resolution Analysis
# ============================================================
#
# PURPOSE:
# We need to select ONE brand for the Hiver assignment.
#
# We already evaluated:
#   1. Support-account volume
#   2. Direct customer interactions
#   3. Distinct customers
#   4. Customer-message diversity
#
# The assignment also asks us to consider whether the brand
# has useful historical support resolutions.
#
# This script measures:
#   - number of customer->brand interactions
#   - unique customer messages
#   - unique historical agent responses
#   - average response length
#   - resolution-response diversity
#
# These are screening metrics, not final performance metrics.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import pandas as pd
import re


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Find the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original Twitter dataset.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "twcs.csv"
)

# Previously generated brand analysis.
BRAND_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brand_conversation_analysis.csv"
)

# Output file for this analysis.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brand_resolution_analysis.csv"
)

# Create the output directory automatically.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. SETTINGS
# ============================================================

# Analyze the strongest 10 candidates.
TOP_N = 10

# Read the large CSV in chunks.
CHUNK_SIZE = 100_000


# ============================================================
# 4. LOAD TOP BRANDS
# ============================================================

brands_df = pd.read_csv(
    BRAND_ANALYSIS_PATH
)

# Select the top candidates according to direct
# customer interactions.
brands_df = (
    brands_df
    .sort_values(
        "direct_customer_interactions",
        ascending=False
    )
    .head(TOP_N)
    .copy()
)

candidate_brands = set(
    brands_df["author_id"]
    .astype(str)
)

print("=" * 70)
print("HIVER SDE - HISTORICAL RESOLUTION ANALYSIS")
print("=" * 70)

print("\nBrands being analyzed:")

for brand in brands_df["author_id"]:
    print(f"- {brand}")


# ============================================================
# 5. PASS 1
# FIND BRAND RESPONSES AND THEIR PARENT CUSTOMER TWEETS
# ============================================================

brand_responses = []

print("\n")
print("=" * 70)
print("PASS 1 - COLLECT BRAND RESPONSES")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
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

    # Keep candidate-brand messages.
    rows = chunk[
        chunk["author_id"]
        .astype(str)
        .isin(candidate_brands)
    ].copy()

    # Keep outbound/support messages only.
    rows = rows[
        rows["inbound"] == False
    ]

    # Only messages that directly respond to another tweet.
    rows = rows[
        rows["in_response_to_tweet_id"].notna()
    ]

    if rows.empty:
        continue

    # Convert parent tweet ID to numeric.
    rows["parent_tweet_id"] = pd.to_numeric(
        rows["in_response_to_tweet_id"],
        errors="coerce"
    )

    # Remove invalid IDs.
    rows = rows[
        rows["parent_tweet_id"].notna()
    ]

    if rows.empty:
        continue

    rows["parent_tweet_id"] = (
        rows["parent_tweet_id"]
        .astype("int64")
    )

    # Store the brand response information.
    brand_responses.append(
        rows[
            [
                "tweet_id",
                "author_id",
                "created_at",
                "text",
                "parent_tweet_id",
            ]
        ]
    )


# Combine all response chunks.
brand_response_df = pd.concat(
    brand_responses,
    ignore_index=True
)

print(
    f"\nBrand responses collected: "
    f"{len(brand_response_df):,}"
)


# ============================================================
# 6. PASS 2
# RETRIEVE CUSTOMER PARENT TWEETS
# ============================================================

parent_ids = set(
    brand_response_df["parent_tweet_id"]
)

customer_rows = []

print("\n")
print("=" * 70)
print("PASS 2 - RETRIEVE CUSTOMER PARENT TWEETS")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
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

    # Retrieve only tweets that are parents of
    # candidate-brand responses.
    rows = chunk[
        chunk["tweet_id"].isin(parent_ids)
    ].copy()

    if rows.empty:
        continue

    # Keep customer/inbound tweets.
    rows = rows[
        rows["inbound"] == True
    ]

    if rows.empty:
        continue

    customer_rows.append(
        rows
    )


customer_df = pd.concat(
    customer_rows,
    ignore_index=True
)


# Rename columns to make the join easier to understand.
customer_df = customer_df.rename(
    columns={
        "tweet_id": "parent_tweet_id",
        "author_id": "customer_author_id",
        "text": "customer_text",
    }
)

print(
    f"\nCustomer parent tweets found: "
    f"{len(customer_df):,}"
)


# ============================================================
# 7. JOIN CUSTOMER MESSAGES WITH HISTORICAL RESPONSES
# ============================================================

conversation_df = brand_response_df.merge(
    customer_df,
    on="parent_tweet_id",
    how="inner"
)


# Remove missing text.
conversation_df = conversation_df[
    conversation_df["customer_text"].notna()
    & conversation_df["text"].notna()
].copy()


# ============================================================
# 8. BASIC TEXT CLEANING
# ============================================================

def normalize_text(text):
    """
    Normalize text for measuring repeated responses.

    This is NOT the final preprocessing pipeline.
    We only want to identify whether historical responses
    are repeated verbatim or contain meaningful variation.
    """

    text = str(text).lower()

    # Remove URLs.
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# Create normalized response text.
conversation_df["normalized_response"] = (
    conversation_df["text"]
    .apply(normalize_text)
)

# Create normalized customer text.
conversation_df["normalized_customer"] = (
    conversation_df["customer_text"]
    .apply(normalize_text)
)


# ============================================================
# 9. CALCULATE RESOLUTION METRICS
# ============================================================

results = []


for brand, group in conversation_df.groupby(
    "author_id"
):

    total_interactions = len(group)

    unique_customers = (
        group["customer_author_id"]
        .nunique()
    )

    unique_customer_messages = (
        group["normalized_customer"]
        .nunique()
    )

    unique_responses = (
        group["normalized_response"]
        .nunique()
    )

    # Average response length.
    average_response_length = (
        group["text"]
        .astype(str)
        .str.len()
        .mean()
    )

    # How many different responses does the brand use
    # per 100 interactions?
    response_diversity_ratio = (
        unique_responses / total_interactions
        if total_interactions > 0
        else 0
    )

    # Most frequent response.
    most_common_response_count = (
        group["normalized_response"]
        .value_counts()
        .iloc[0]
        if total_interactions > 0
        else 0
    )

    # Percentage of interactions using the most common
    # response template.
    most_common_response_ratio = (
        most_common_response_count
        / total_interactions
        if total_interactions > 0
        else 0
    )

    results.append(
        {
            "brand":
                brand,

            "historical_interactions":
                total_interactions,

            "distinct_customers":
                unique_customers,

            "unique_customer_messages":
                unique_customer_messages,

            "unique_historical_responses":
                unique_responses,

            "response_diversity_ratio":
                round(
                    response_diversity_ratio,
                    4
                ),

            "avg_response_length":
                round(
                    average_response_length,
                    2
                ),

            "most_common_response_ratio":
                round(
                    most_common_response_ratio,
                    4
                ),
        }
    )


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

resolution_df = pd.DataFrame(
    results
)

# Sort by number of unique historical responses.
resolution_df = (
    resolution_df
    .sort_values(
        "unique_historical_responses",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HISTORICAL RESOLUTION DIVERSITY")
print("=" * 70)

print(
    resolution_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

resolution_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nOutput exists: "
    f"{OUTPUT_PATH.exists()}"
)