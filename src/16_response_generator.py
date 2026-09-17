"""
Grounded Response Generator
----------------------------

Purpose:
    Generate a customer-support reply using historically similar
    AmazonHelp conversations retrieved from the FAISS index.

Flow:
    Customer message
        ↓
    FAISS historical retrieval
        ↓
    Top-K similar historical conversations
        ↓
    Gemini
        ↓
    Grounded draft response

Important:
    The model is instructed NOT to invent policies, refunds,
    delivery promises, compensation, or other facts that are
    not supported by the retrieved historical responses.
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import os

import faiss
import pandas as pd

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PROJECT PATH
# ============================================================

# The script is inside the "src" folder.
# parents[1] gives us the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load the Gemini API key from the project's .env file.
load_dotenv(
    PROJECT_ROOT / ".env"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# Stop if the API key is missing.
if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

# Create the current Google GenAI client.
client = genai.Client(
    api_key=GEMINI_API_KEY
)

# Use the Gemini model that was successfully tested earlier.
MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# RETRIEVAL FILES
# ============================================================

# FAISS vector index created by script 15.
INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_faiss.index"
)

# Metadata corresponding to each FAISS vector.
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazonhelp_retrieval_metadata.csv"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

# This MUST match the embedding model used to create the FAISS
# index. Otherwise the vectors would not be compatible.
EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# Retrieve the five most similar historical conversations.
TOP_K = 5


# ============================================================
# CHECK FILES
# ============================================================

if not INDEX_FILE.exists():

    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_FILE}"
    )


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("=" * 70)
print("LOADING HISTORICAL RESOLUTION INDEX")
print("=" * 70)

# Read the saved FAISS index.
index = faiss.read_index(
    str(INDEX_FILE)
)

print(
    f"Indexed historical examples: {index.ntotal}"
)


# ============================================================
# LOAD RETRIEVAL METADATA
# ============================================================

# The metadata rows must have exactly the same ordering as the
# vectors stored in FAISS.
metadata = pd.read_csv(
    METADATA_FILE
)

print(
    f"Metadata rows: {len(metadata)}"
)


# ============================================================
# CHECK INDEX / METADATA ALIGNMENT
# ============================================================

# Every FAISS vector must have one corresponding metadata row.
if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata row counts do not match. "
        f"Index={index.ntotal}, Metadata={len(metadata)}"
    )


# ============================================================
# IDENTIFY TEXT COLUMNS
# ============================================================

# Find the customer-message column.
possible_customer_columns = [
    "customer_message",
    "customer_text",
    "text",
]

customer_column = None

for column in possible_customer_columns:

    if column in metadata.columns:

        customer_column = column
        break


# Find the historical agent-response column.
possible_response_columns = [
    "agent_response",
    "response",
    "response_text",
]

response_column = None

for column in possible_response_columns:

    if column in metadata.columns:

        response_column = column
        break


# Stop if either required column is unavailable.
if customer_column is None:

    raise ValueError(
        "Could not find the historical customer-message column."
    )


if response_column is None:

    raise ValueError(
        "Could not find the historical agent-response column."
    )


print(
    f"Customer column: {customer_column}"
)

print(
    f"Agent response column: {response_column}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

# Load exactly the same model used to create the FAISS index.
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# RETRIEVE HISTORICAL RESOLUTIONS
# ============================================================

def retrieve_historical_resolutions(
    customer_message: str,
    top_k: int = TOP_K,
):
    """
    Retrieve the most semantically similar historical
    AmazonHelp conversations.

    Returns:
        List of dictionaries containing:
            rank
            similarity
            customer_message
            agent_response
    """

    # Create an embedding for the new customer message.
    query_embedding = embedding_model.encode(
        [customer_message],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # FAISS expects float32 vectors.
    query_embedding = query_embedding.astype(
        "float32"
    )

    # Search for the nearest historical examples.
    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    retrieved = []

    # Process every retrieved result.
    for rank in range(top_k):

        row_index = int(
            indices[0][rank]
        )

        similarity = float(
            scores[0][rank]
        )

        # Skip invalid FAISS indices.
        if row_index < 0:
            continue

        # Get the corresponding metadata row.
        row = metadata.iloc[
            row_index
        ]

        retrieved.append(
            {
                "rank": rank + 1,
                "similarity": similarity,
                "customer_message": str(
                    row[customer_column]
                ),
                "agent_response": str(
                    row[response_column]
                ),
            }
        )

    return retrieved


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_generation_prompt(
    customer_message: str,
    retrieved_resolutions: list,
):
    """
    Build the grounded response-generation prompt.

    Gemini is instructed to use the historical examples as
    evidence and avoid unsupported promises.
    """

    historical_section = []

    # Format every retrieved historical conversation.
    for item in retrieved_resolutions:

        historical_section.append(
            f"""
HISTORICAL EXAMPLE {item["rank"]}
Similarity: {item["similarity"]:.4f}

Previous customer:
{item["customer_message"]}

Historical Amazon support response:
{item["agent_response"]}
""".strip()
        )

    historical_text = "\n\n".join(
        historical_section
    )

    prompt = f"""
You are an Amazon customer-support response drafting assistant.

Your job is to draft a helpful response to the new customer message.

NEW CUSTOMER MESSAGE:
{customer_message}

Below are historically similar AmazonHelp conversations and the
responses that Amazon support actually gave.

HISTORICAL EVIDENCE:
{historical_text}

GROUNDING RULES:
1. Use the historical responses as evidence for the draft.
2. Do not invent Amazon policies, refunds, compensation, delivery
   dates, contact numbers, or guarantees.
3. Do not claim an action was taken when there is no evidence that
   it was taken.
4. If the historical examples suggest contacting another support
   channel, you may recommend that only when supported by the
   examples.
5. Keep the response concise and professional.
6. Acknowledge the customer's problem.
7. When the historical evidence does not provide a safe concrete
   resolution, give a cautious next step instead of inventing one.

Return ONLY the customer-facing response.
Do not mention:
- embeddings
- FAISS
- historical examples
- similarity scores
- AI
- internal systems

NEW RESPONSE:
"""

    return prompt.strip()


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    customer_message: str,
):
    """
    Retrieve similar cases and generate a grounded response.

    Returns:
        response_text
        retrieved_examples
    """

    # Retrieve historically similar support cases.
    retrieved = retrieve_historical_resolutions(
        customer_message
    )

    # Build the grounded prompt.
    prompt = build_generation_prompt(
        customer_message,
        retrieved,
    )

    # Call Gemini.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    # Read generated text.
    response_text = getattr(
        response,
        "text",
        None,
    )

    if not response_text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return (
        response_text.strip(),
        retrieved,
    )


# ============================================================
# TEST EXAMPLES
# ============================================================

# These are development examples only.
# They help us check whether retrieval + generation are working.
TEST_MESSAGES = [
    (
        "My package says delivered but I never received it."
    ),
    (
        "I returned my item but still haven't received my refund."
    ),
    (
        "I received the wrong product in my order."
    ),
    (
        "Why can't I watch this show on Prime Video?"
    ),
]


# ============================================================
# RUN TESTS
# ============================================================

print("\n")
print("=" * 70)
print("TESTING GROUNDED RESPONSE GENERATION")
print("=" * 70)


for test_number, customer_message in enumerate(
    TEST_MESSAGES,
    start=1,
):

    print("\n")
    print("-" * 70)

    print(
        f"TEST {test_number}"
    )

    print("-" * 70)

    print(
        "\nCustomer:"
    )

    print(
        customer_message
    )

    # Generate a response from the retrieved historical cases.
    generated_response, retrieved = (
        generate_response(
            customer_message
        )
    )

    # Show the retrieved evidence.
    print(
        "\nTop retrieved historical cases:"
    )

    for item in retrieved:

        print(
            f"\nRank {item['rank']} "
            f"| Similarity={item['similarity']:.4f}"
        )

        print(
            f"Customer: "
            f"{item['customer_message'][:200]}"
        )

        print(
            f"Response: "
            f"{item['agent_response'][:200]}"
        )

    # Show the generated customer response.
    print(
        "\nGenerated response:"
    )

    print(
        generated_response
    )


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("RESPONSE GENERATION TEST COMPLETE")
print("=" * 70)