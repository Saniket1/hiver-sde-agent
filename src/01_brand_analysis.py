from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# __file__ = ...\hiver-sde-agent\src\01_brand_analysis.py
# parent      = ...\hiver-sde-agent\src
# parent.parent = ...\hiver-sde-agent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "brand_candidates.csv"

# Make sure the output directory exists.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure twcs.csv is inside:\n"
        "data/raw/\n"
    )

print("=" * 70)
print("HIVER SDE ASSIGNMENT - BRAND ANALYSIS")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset      : {DATA_PATH}")
print(f"Output       : {OUTPUT_PATH}")

print(f"\nDataset size: {DATA_PATH.stat().st_size / (1024 ** 2):.2f} MB")


# ============================================================
# 3. COLUMNS NEEDED
# ============================================================

usecols = [
    "tweet_id",
    "author_id",
    "inbound",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


# ============================================================
# 4. PROCESS DATASET IN CHUNKS
# ============================================================

# We use chunks because the dataset is approximately 516 MB.
CHUNK_SIZE = 100_000

# Dictionary to accumulate statistics for every author.
stats = {}


print("\nStarting brand analysis...\n")


for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):
    print(f"Processing chunk {chunk_number}...")

    # --------------------------------------------------------
    # Inbound = customer-side tweet
    # Outbound = brand/support-side tweet
    # --------------------------------------------------------

    inbound = chunk[chunk["inbound"] == True]
    outbound = chunk[chunk["inbound"] == False]

    # Count all messages by author.
    total_counts = chunk.groupby("author_id").size()

    # Count inbound messages by author.
    inbound_counts = inbound.groupby("author_id").size()

    # Count outbound messages by author.
    outbound_counts = outbound.groupby("author_id").size()

    # Update statistics.
    for author, count in total_counts.items():

        if author not in stats:
            stats[author] = {
                "total_messages": 0,
                "inbound_messages": 0,
                "outbound_messages": 0,
            }

        stats[author]["total_messages"] += int(count)

    for author, count in inbound_counts.items():

        if author not in stats:
            stats[author] = {
                "total_messages": 0,
                "inbound_messages": 0,
                "outbound_messages": 0,
            }

        stats[author]["inbound_messages"] += int(count)

    for author, count in outbound_counts.items():

        if author not in stats:
            stats[author] = {
                "total_messages": 0,
                "inbound_messages": 0,
                "outbound_messages": 0,
            }

        stats[author]["outbound_messages"] += int(count)


# ============================================================
# 5. CREATE RESULTS DATAFRAME
# ============================================================

results = []

for author, values in stats.items():

    results.append(
        {
            "author_id": author,
            "total_messages": values["total_messages"],
            "inbound_messages": values["inbound_messages"],
            "outbound_messages": values["outbound_messages"],
        }
    )


brand_df = pd.DataFrame(results)


# ============================================================
# 6. IDENTIFY CANDIDATE SUPPORT/BRAND ACCOUNTS
# ============================================================

# We are interested in accounts that have outbound messages,
# because the support/brand side sends outbound replies.

brand_df = brand_df[
    brand_df["outbound_messages"] > 0
].copy()


# Sort primarily by outbound support volume.
brand_df = brand_df.sort_values(
    by="outbound_messages",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 7. PRINT TOP CANDIDATES
# ============================================================

print("\n")
print("=" * 70)
print("TOP 50 CANDIDATE SUPPORT / BRAND ACCOUNTS")
print("=" * 70)

print(
    brand_df.head(50).to_string(index=False)
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

brand_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nSaved results to:")
print(OUTPUT_PATH)

print(
    f"\nNumber of candidate accounts identified: {len(brand_df):,}"
)

print(
    f"Output file exists: {OUTPUT_PATH.exists()}"
)