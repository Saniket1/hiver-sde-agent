from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "brand_candidates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "brand_conversation_analysis.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

# Analyze the top candidates rather than all 108 initially.
TOP_N = 30

CHUNK_SIZE = 100_000


# ============================================================
# 3. LOAD CANDIDATE BRANDS
# ============================================================

if not CANDIDATES_PATH.exists():
    raise FileNotFoundError(
        f"Candidate file not found:\n{CANDIDATES_PATH}"
    )

candidates = pd.read_csv(CANDIDATES_PATH)

candidate_brands = (
    candidates
    .sort_values("outbound_messages", ascending=False)
    .head(TOP_N)["author_id"]
    .astype(str)
    .tolist()
)

candidate_set = set(candidate_brands)

print("=" * 70)
print("HIVER SDE - BRAND / CONVERSATION ANALYSIS")
print("=" * 70)

print(f"\nAnalyzing top {TOP_N} candidate brands:")
print(candidate_brands)


# ============================================================
# 4. FIRST PASS
#
# Find brand tweets that directly respond to another tweet.
# Store the parent tweet ID so we can identify the customer.
# ============================================================

brand_responses = []

print("\n")
print("=" * 70)
print("PASS 1 - FIND BRAND RESPONSE LINKS")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "in_response_to_tweet_id",
        ],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):
    print(f"Processing chunk {chunk_number}...")

    # Only candidate brand/support accounts.
    brand_rows = chunk[
        (chunk["author_id"].astype(str).isin(candidate_set))
        & (chunk["inbound"] == False)
        & (chunk["in_response_to_tweet_id"].notna())
    ].copy()

    if not brand_rows.empty:
        brand_responses.append(
            brand_rows[
                [
                    "tweet_id",
                    "author_id",
                    "in_response_to_tweet_id",
                ]
            ]
        )


if not brand_responses:
    raise RuntimeError(
        "No candidate brand responses with parent tweet links were found."
    )


brand_responses_df = pd.concat(
    brand_responses,
    ignore_index=True
)


# Convert parent IDs into integer-compatible values.
brand_responses_df["parent_tweet_id"] = pd.to_numeric(
    brand_responses_df["in_response_to_tweet_id"],
    errors="coerce"
)

brand_responses_df = brand_responses_df[
    brand_responses_df["parent_tweet_id"].notna()
].copy()

brand_responses_df["parent_tweet_id"] = (
    brand_responses_df["parent_tweet_id"]
    .astype("int64")
)

print(
    f"\nCandidate brand responses found: "
    f"{len(brand_responses_df):,}"
)

parent_ids = set(
    brand_responses_df["parent_tweet_id"].tolist()
)

print(
    f"Unique parent tweet IDs to inspect: "
    f"{len(parent_ids):,}"
)


# ============================================================
# 5. SECOND PASS
#
# Retrieve the parent tweets and determine whether they were
# customer/inbound messages.
# ============================================================

parent_rows = []

print("\n")
print("=" * 70)
print("PASS 2 - IDENTIFY CUSTOMER PARENT TWEETS")
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
        ],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):
    print(f"Processing chunk {chunk_number}...")

    matching = chunk[
        chunk["tweet_id"].isin(parent_ids)
    ].copy()

    if not matching.empty:
        parent_rows.append(matching)


if parent_rows:
    parent_df = pd.concat(
        parent_rows,
        ignore_index=True
    )
else:
    parent_df = pd.DataFrame(
        columns=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
        ]
    )


print(
    f"\nParent tweets found: {len(parent_df):,}"
)


# ============================================================
# 6. JOIN BRAND RESPONSES TO PARENT TWEETS
# ============================================================

parent_lookup = parent_df.rename(
    columns={
        "tweet_id": "parent_tweet_id",
        "author_id": "customer_author_id",
        "inbound": "parent_inbound",
        "created_at": "customer_created_at",
        "text": "customer_text",
    }
)

merged = brand_responses_df.merge(
    parent_lookup,
    on="parent_tweet_id",
    how="left"
)


# ============================================================
# 7. KEEP ONLY BRAND RESPONSES TO INBOUND/CUSTOMER TWEETS
# ============================================================

customer_interactions = merged[
    merged["parent_inbound"] == True
].copy()


# ============================================================
# 8. AGGREGATE BRAND METRICS
# ============================================================

summary_rows = []

for brand, group in customer_interactions.groupby(
    "author_id"
):

    summary_rows.append(
        {
            "brand": brand,

            "direct_customer_interactions":
                len(group),

            "distinct_customers":
                group["customer_author_id"].nunique(),

            "unique_customer_tweets":
                group["parent_tweet_id"].nunique(),

            "customer_text_available":
                group["customer_text"].notna().sum(),

            "response_link_coverage":
                round(
                    group["customer_text"].notna().mean(),
                    4
                ),
        }
    )


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 9. MERGE WITH ORIGINAL VOLUME DATA
# ============================================================

volume_df = candidates[
    candidates["author_id"].isin(candidate_brands)
].copy()

final_df = volume_df.merge(
    summary_df,
    left_on="author_id",
    right_on="brand",
    how="left"
)

final_df["direct_customer_interactions"] = (
    final_df["direct_customer_interactions"]
    .fillna(0)
    .astype(int)
)

final_df["distinct_customers"] = (
    final_df["distinct_customers"]
    .fillna(0)
    .astype(int)
)

final_df["unique_customer_tweets"] = (
    final_df["unique_customer_tweets"]
    .fillna(0)
    .astype(int)
)

final_df["customer_text_available"] = (
    final_df["customer_text_available"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 10. RANK BY DIRECT CUSTOMER INTERACTIONS
# ============================================================

final_df = final_df.sort_values(
    "direct_customer_interactions",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TOP BRANDS BY DIRECT CUSTOMER INTERACTIONS")
print("=" * 70)

display_columns = [
    "author_id",
    "outbound_messages",
    "direct_customer_interactions",
    "distinct_customers",
    "unique_customer_tweets",
    "response_link_coverage",
]

print(
    final_df[display_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print(
    f"\nOutput exists: {OUTPUT_PATH.exists()}"
)