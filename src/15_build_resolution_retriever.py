"""
Historical Resolution Retriever
--------------------------------

Purpose:
    Build a semantic retrieval index over historical AmazonHelp
    customer messages and their real agent responses.

Why:
    The Hiver agent should ground suggested replies in historical
    support resolutions instead of generating responses from
    scratch.

Pipeline:
    1. Load AmazonHelp conversation history.
    2. Clean customer messages and agent responses.
    3. Create sentence embeddings for historical customer messages.
    4. Build a FAISS similarity-search index.
    5. Save the index and metadata.
    6. Test retrieval using a few example queries.
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT FILE
# ============================================================

# Historical AmazonHelp customer-response conversations
# created earlier in the project.
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_conversations.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

# FAISS vector index.
INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_faiss.index"
)

# Metadata corresponding to each vector in the FAISS index.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_retrieval_metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of historical conversations to retrieve for each query.
TOP_K = 5

# Use a lightweight sentence-transformer model.
MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Historical conversation file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("=" * 70)
print("LOADING AMAZONHELP HISTORICAL CONVERSATIONS")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# SHOW COLUMNS
# ============================================================

print("\nAvailable columns:")

print(
    df.columns.tolist()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

# The earlier conversation-building script is expected to
# contain customer and agent text. We check for the names
# explicitly instead of silently assuming them.

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


# Find the first matching customer column.
customer_column = None

for column in possible_customer_columns:

    if column in df.columns:

        customer_column = column
        break


# Find the first matching response column.
response_column = None

for column in possible_response_columns:

    if column in df.columns:

        response_column = column
        break


# Stop if the expected data is missing.
if customer_column is None:

    raise ValueError(
        "Could not find a customer-message column. "
        f"Available columns: {df.columns.tolist()}"
    )


if response_column is None:

    raise ValueError(
        "Could not find an agent-response column. "
        f"Available columns: {df.columns.tolist()}"
    )


print(
    f"\nCustomer column: {customer_column}"
)

print(
    f"Agent response column: {response_column}"
)


# ============================================================
# CLEAN TEXT
# ============================================================

# Keep only rows where both customer and response text exist.
df = df.dropna(
    subset=[
        customer_column,
        response_column,
    ]
).copy()


# Convert both columns to strings.
df[customer_column] = (
    df[customer_column]
    .astype(str)
    .str.strip()
)

df[response_column] = (
    df[response_column]
    .astype(str)
    .str.strip()
)


# Remove empty customer messages.
df = df[
    df[customer_column] != ""
].copy()


# Remove empty agent responses.
df = df[
    df[response_column] != ""
].copy()


# Reset dataframe index so FAISS row numbers map cleanly to
# metadata rows.
df = df.reset_index(
    drop=True
)


print(
    f"\nUsable conversation pairs: {len(df)}"
)


# ============================================================
# OPTIONAL SUBSAMPLE
# ============================================================

# The full AmazonHelp dataset contains a large number of
# historical interactions. For this development index we use
# a manageable subset.
#
# We use a deterministic sample so that the experiment can be
# reproduced.
MAX_RECORDS = 50000

if len(df) > MAX_RECORDS:

    df = df.sample(
        n=MAX_RECORDS,
        random_state=42,
    ).reset_index(
        drop=True
    )

    print(
        f"Using deterministic sample: {len(df)} rows"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embedding_model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating historical customer-message embeddings...")

# Convert each historical customer message into a semantic
# vector.
embeddings = embedding_model.encode(
    df[customer_column].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING FAISS INDEX")
print("=" * 70)

# Convert to float32 because FAISS expects float32 vectors.
embeddings = embeddings.astype(
    "float32"
)


# Since vectors are normalized, inner product is equivalent
# to cosine similarity.
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


# Add all historical vectors to the index.
index.add(
    embeddings
)


print(
    f"Embedding dimension: {dimension}"
)

print(
    f"Indexed vectors: {index.ntotal}"
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

# Save the vector index to disk.
faiss.write_index(
    index,
    str(INDEX_FILE),
)


# ============================================================
# SAVE METADATA
# ============================================================

# Keep the information needed to interpret every retrieved
# vector.
metadata_columns = [
    customer_column,
    response_column,
]

# Include useful identifier columns when they exist.
for column in [
    "conversation_id",
    "timestamp",
    "customer_id",
    "tweet_id",
]:

    if (
        column in df.columns
        and column not in metadata_columns
    ):

        metadata_columns.append(
            column
        )


metadata = df[
    metadata_columns
].copy()


# Save metadata using the exact same row order as the FAISS
# vectors.
metadata.to_csv(
    METADATA_FILE,
    index=False,
)


# ============================================================
# TEST RETRIEVAL
# ============================================================

print("\n")
print("=" * 70)
print("TESTING RETRIEVAL")
print("=" * 70)


# Example customer-support queries.
test_queries = [
    "Where is my package? It says delivered but I never received it.",
    "My refund has not arrived yet.",
    "I received the wrong item.",
    "Why is my Prime Video show unavailable?",
]


# Convert the test queries into embeddings.
query_embeddings = embedding_model.encode(
    test_queries,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(
    "float32"
)


# Search the FAISS index.
scores, indices = index.search(
    query_embeddings,
    TOP_K,
)


# Display retrieved examples.
for query_number, query in enumerate(
    test_queries
):

    print("\n")
    print("-" * 70)

    print(
        f"QUERY: {query}"
    )

    print("-" * 70)

    for rank in range(TOP_K):

        row_index = int(
            indices[
                query_number,
                rank,
            ]
        )

        score = float(
            scores[
                query_number,
                rank,
            ]
        )

        retrieved_customer = metadata.iloc[
            row_index
        ][customer_column]

        retrieved_response = metadata.iloc[
            row_index
        ][response_column]

        print(
            f"\nRank {rank + 1}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            f"Customer: {retrieved_customer[:250]}"
        )

        print(
            f"Agent response: {retrieved_response[:250]}"
        )


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVAL INDEX COMPLETE")
print("=" * 70)

print(
    "FAISS index:"
)

print(
    INDEX_FILE
)

print(
    "\nMetadata:"
)

print(
    METADATA_FILE
)