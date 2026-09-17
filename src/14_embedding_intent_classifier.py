"""
Sentence-Embedding Intent Classifier
-------------------------------------

Purpose:
    Build a semantic intent classifier using sentence-transformer
    embeddings and nearest-centroid classification.

Method:
    1. Load the human-verified golden set.
    2. Split the data into train and test sets.
    3. Convert customer messages into sentence embeddings.
    4. Build one centroid embedding for each intent.
    5. Assign each test message to the closest intent centroid.
    6. Evaluate accuracy, precision, recall, and F1.
    7. Save predictions for later error analysis.

Important:
    Only the training portion is used to create intent centroids.
    Test examples are NOT used to build the centroids.
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATH
# ============================================================

# Find the project root automatically.
# This script is inside the "src" folder, so parents[1]
# points to the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# FILE PATHS
# ============================================================

# Final human-verified golden set.
GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

# Output file containing semantic-classifier predictions.
RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "embedding_predictions.csv"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

# Create the results directory if it does not already exist.
RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK GOLDEN SET
# ============================================================

# Stop with a clear error if the golden file does not exist.
if not GOLDEN_FILE.exists():

    raise FileNotFoundError(
        f"Golden set not found:\n{GOLDEN_FILE}"
    )


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("=" * 70)
print("LOADING GOLDEN SET")
print("=" * 70)

# Load the human-verified data.
df = pd.read_csv(
    GOLDEN_FILE
)

print(
    f"Loaded rows: {len(df)}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

# These columns are required for intent classification.
required_columns = {
    "customer_message",
    "true_intent",
}

# Find any missing required columns.
missing_columns = (
    required_columns
    - set(df.columns)
)

# Stop if the required columns are missing.
if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

# Remove rows where either the message or human label is missing.
df = df.dropna(
    subset=[
        "customer_message",
        "true_intent",
    ]
).copy()

# Convert customer messages into clean strings.
df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

# Remove completely empty messages.
df = df[
    df["customer_message"] != ""
].copy()

print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# INPUTS AND TARGET
# ============================================================

# X contains the customer messages.
X = df["customer_message"]

# y contains the final human-verified intent labels.
y = df["true_intent"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Use the same 75/25 split as the TF-IDF baseline.
#
# Stratification keeps the class distribution as balanced as
# possible between train and test.
#
# random_state=42 makes the experiment reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


print("\n")
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training examples: {len(X_train)}"
)

print(
    f"Test examples: {len(X_test)}"
)

print(
    f"Number of intents: {y.nunique()}"
)


# ============================================================
# LOAD SENTENCE-TRANSFORMER MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

# all-MiniLM-L6-v2 is a lightweight sentence embedding model.
#
# It converts each customer message into a numeric vector that
# represents semantic meaning rather than just individual words.
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE TRAINING EMBEDDINGS
# ============================================================

print("\nCreating training embeddings...")

# Convert each training customer message into an embedding.
#
# normalize_embeddings=True means every vector is normalized,
# which makes cosine similarity easy to calculate using a dot
# product later.
train_embeddings = embedding_model.encode(
    X_train.tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# ============================================================
# CREATE TEST EMBEDDINGS
# ============================================================

print("\nCreating test embeddings...")

# Convert test messages into embeddings.
#
# These embeddings are used only for prediction/evaluation.
test_embeddings = embedding_model.encode(
    X_test.tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# ============================================================
# BUILD INTENT CENTROIDS
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING INTENT CENTROIDS")
print("=" * 70)

# Dictionary where:
#     key   = intent name
#     value = centroid embedding for that intent
centroids = {}


# Process every intent present in the training set.
for intent in sorted(
    y_train.unique()
):

    # Boolean mask identifying training rows belonging to this
    # particular intent.
    mask = (
        y_train.values == intent
    )

    # Select the embeddings for that intent.
    intent_embeddings = (
        train_embeddings[mask]
    )

    # Calculate the average embedding.
    #
    # This average vector represents the semantic "center" of
    # the intent.
    centroid = np.mean(
        intent_embeddings,
        axis=0,
    )

    # Calculate the vector magnitude.
    centroid_norm = np.linalg.norm(
        centroid
    )

    # Normalize the centroid if it is not a zero vector.
    if centroid_norm > 0:

        centroid = (
            centroid
            / centroid_norm
        )

    # Save the centroid.
    centroids[intent] = centroid

    print(
        f"{intent}: "
        f"{len(intent_embeddings)} training examples"
    )


# ============================================================
# PREDICT TEST EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTING TEST EXAMPLES")
print("=" * 70)

# Store predicted intent for every test example.
predictions = []

# Store similarity score for every prediction.
similarity_scores = []


# Process every test embedding.
for embedding in test_embeddings:

    # Start with no best intent.
    best_intent = None

    # Start with the lowest possible cosine similarity.
    best_score = -1.0

    # Compare the message against every intent centroid.
    for intent, centroid in centroids.items():

        # Because both the message embedding and centroid are
        # normalized, their dot product is cosine similarity.
        score = float(
            np.dot(
                embedding,
                centroid,
            )
        )

        # Keep the most similar intent.
        if score > best_score:

            best_score = score
            best_intent = intent

    # Store the predicted intent.
    predictions.append(
        best_intent
    )

    # Store the similarity score.
    similarity_scores.append(
        best_score
    )


# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EMBEDDING CLASSIFIER RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

# Display precision, recall, F1, and support for every intent.
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Use the complete intent list from the dataset.
labels = sorted(
    y.unique()
)

# Calculate the confusion matrix.
cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)


print("\nConfusion matrix:")

# Convert the matrix into a readable pandas table.
print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

# Build a dataframe containing:
#     - original customer message
#     - human-verified intent
#     - predicted intent
#     - semantic similarity score
prediction_df = pd.DataFrame(
    {
        "customer_message": X_test.values,
        "true_intent": y_test.values,
        "predicted_intent": predictions,
        "similarity_score": similarity_scores,
    }
)


# Save predictions to the results folder.
prediction_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 70)
print("EMBEDDING CLASSIFIER COMPLETE")
print("=" * 70)

print(
    "Predictions saved to:"
)

print(
    RESULT_FILE
)