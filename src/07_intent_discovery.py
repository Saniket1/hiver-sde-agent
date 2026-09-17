# ============================================================
# Hiver SDE Intern Assignment
# Step 10 - Intent Discovery Exploration
# ============================================================
#
# PURPOSE:
# Explore the 500 AmazonHelp customer messages that we sampled
# from the processed conversation dataset.
#
# IMPORTANT:
# The clusters generated here are NOT the final intent labels.
#
# They are only an exploratory tool to help us discover
# recurring customer-support problem areas.
#
# Workflow:
#
#   500 customer messages
#          ↓
#   TF-IDF representation
#          ↓
#   K-Means clustering
#          ↓
#   Important cluster terms
#          ↓
#   Representative examples
#          ↓
#   Human review
#          ↓
#   Final 8-12 intent taxonomy
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# Path helps us create reliable file paths on Windows.
from pathlib import Path

# pandas is used for loading and manipulating our sample data.
import pandas as pd

# NumPy is used for numerical calculations.
import numpy as np

# TF-IDF converts customer messages into numerical vectors.
from sklearn.feature_extraction.text import TfidfVectorizer

# K-Means is used only for exploratory clustering.
from sklearn.cluster import KMeans


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

# Current file:
#
# hiver-sde-agent/
#     src/
#         07_intent_discovery.py
#
# parent       -> src
# parent.parent -> hiver-sde-agent
#
# Therefore PROJECT_ROOT is our main project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Our 500-message discovery sample created in Step 9.
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_discovery_sample.csv"
)


# Output file containing representative examples from
# each exploratory cluster.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intent_discovery_clusters.csv"
)


# Create the output directory if it doesn't already exist.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. SETTINGS
# ============================================================

# The assignment guidance recommends approximately 8-12 intents.
#
# We use 10 clusters as an exploratory starting point.
# These are NOT our final 10 intents.
NUMBER_OF_CLUSTERS = 10


# Number of representative examples displayed for each cluster.
EXAMPLES_PER_CLUSTER = 10


# ============================================================
# 4. CHECK INPUT FILE
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nIntent discovery sample was not found:\n"
        f"{INPUT_PATH}\n\n"
        "Run 06_validate_conversations.py first."
    )


# ============================================================
# 5. LOAD CUSTOMER MESSAGES
# ============================================================

print("=" * 70)
print("HIVER SDE - INTENT DISCOVERY")
print("=" * 70)

print(
    f"\nLoading intent discovery sample:\n"
    f"{INPUT_PATH}"
)


# Read the 500-row discovery sample.
df = pd.read_csv(
    INPUT_PATH
)


# Make sure customer messages are treated as strings.
df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove completely empty customer messages.
df = df[
    df["customer_message"].ne("")
].copy()


# Reset index so our later indexing is straightforward.
df = df.reset_index(
    drop=True
)


print(
    f"\nCustomer messages available: "
    f"{len(df):,}"
)


# ============================================================
# 6. VERIFY ENOUGH DATA FOR CLUSTERING
# ============================================================

if len(df) < NUMBER_OF_CLUSTERS:

    raise ValueError(
        f"Only {len(df)} messages are available, but "
        f"{NUMBER_OF_CLUSTERS} clusters were requested."
    )


# ============================================================
# 7. CREATE TF-IDF REPRESENTATION
# ============================================================
#
# TF-IDF converts text into numerical features.
#
# We use:
#
#   unigram  -> "refund"
#   bigram   -> "refund request"
#
# Bigrams are useful because customer intent is often expressed
# through short phrases rather than individual words.
# ============================================================

print("\n")
print("=" * 70)
print("CREATING TF-IDF REPRESENTATION")
print("=" * 70)


vectorizer = TfidfVectorizer(
    # Treat uppercase/lowercase words the same.
    lowercase=True,

    # Remove common English stop words.
    stop_words="english",

    # Use both single words and two-word phrases.
    ngram_range=(1, 2),

    # Ignore features appearing in only one document.
    min_df=2,

    # Prevent the feature matrix from becoming unnecessarily
    # large.
    max_features=10_000,
)


# Convert customer messages into the TF-IDF matrix.
X = vectorizer.fit_transform(
    df["customer_message"]
)


print(
    f"TF-IDF matrix shape: {X.shape}"
)


# ============================================================
# 8. RUN K-MEANS CLUSTERING
# ============================================================
#
# K-Means groups messages based on similarity in their TF-IDF
# representation.
#
# IMPORTANT:
# These clusters are ONLY exploratory.
# We will manually review them before creating the final
# intent taxonomy.
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING K-MEANS")
print("=" * 70)


kmeans = KMeans(
    # Number of exploratory groups.
    n_clusters=NUMBER_OF_CLUSTERS,

    # Fixed seed makes the experiment reproducible.
    random_state=42,

    # Multiple initializations make K-Means more stable.
    n_init=10,
)


# Assign every customer message to a cluster.
df["cluster"] = kmeans.fit_predict(
    X
)


# ============================================================
# 9. GET TF-IDF FEATURE NAMES
# ============================================================

# These are the actual terms represented by columns in X.
feature_names = vectorizer.get_feature_names_out()


# ============================================================
# 10. PRINT CLUSTER OVERVIEW
# ============================================================

print("\n")
print("=" * 70)
print("CLUSTER OVERVIEW")
print("=" * 70)


for cluster_id in range(
    NUMBER_OF_CLUSTERS
):

    # Number of examples belonging to this cluster.
    cluster_size = int(
        (df["cluster"] == cluster_id).sum()
    )


    # K-Means centroid for this cluster.
    centroid = (
        kmeans.cluster_centers_[cluster_id]
    )


    # Find the features with the highest centroid weights.
    top_indices = np.argsort(
        centroid
    )[-10:][::-1]


    # Convert feature indexes into actual words/phrases.
    top_terms = [
        feature_names[index]
        for index in top_indices
    ]


    print(
        f"\nCluster {cluster_id}"
    )

    print(
        f"Size: {cluster_size}"
    )

    print(
        "Top terms:"
    )

    print(
        ", ".join(top_terms)
    )


# ============================================================
# 11. FIND REPRESENTATIVE EXAMPLES
# ============================================================
#
# For each cluster, we want examples that are closest to the
# cluster center.
#
# These examples help us understand what the cluster is about.
#
# IMPORTANT:
# We are NOT using these clusters as final intent labels.
# ============================================================

print("\n")
print("=" * 70)
print("REPRESENTATIVE CUSTOMER EXAMPLES")
print("=" * 70)


# This list will hold rows that we later save to CSV.
representative_rows = []


for cluster_id in range(
    NUMBER_OF_CLUSTERS
):

    print("\n")
    print("-" * 70)
    print(
        f"CLUSTER {cluster_id}"
    )
    print("-" * 70)


    # --------------------------------------------------------
    # 11.1 Find rows belonging to this cluster
    # --------------------------------------------------------

    cluster_mask = (
        df["cluster"] == cluster_id
    )


    # Get the DataFrame positions belonging to this cluster.
    cluster_positions = np.flatnonzero(
        cluster_mask.to_numpy()
    )


    # --------------------------------------------------------
    # 11.2 Get their TF-IDF vectors
    # --------------------------------------------------------

    cluster_vectors = X[
        cluster_positions
    ]


    # --------------------------------------------------------
    # 11.3 Convert sparse matrix to NumPy array
    # --------------------------------------------------------
    #
    # This is the important correction from the previous code.
    #
    # We explicitly convert to a normal NumPy array before
    # performing element-wise subtraction and squaring.
    #
    # Without this conversion, NumPy may treat "** 2" as
    # matrix power rather than element-wise squaring.
    # --------------------------------------------------------

    cluster_vectors_dense = (
        cluster_vectors.toarray()
    )


    # --------------------------------------------------------
    # 11.4 Get cluster centroid
    # --------------------------------------------------------

    centroid = (
        kmeans.cluster_centers_[cluster_id]
    )


    # --------------------------------------------------------
    # 11.5 Calculate squared Euclidean distance
    # --------------------------------------------------------
    #
    # For each message:
    #
    # distance =
    # sum((message_vector - centroid)^2)
    #
    # Smaller distance means the message is closer to the
    # center of the cluster.
    # --------------------------------------------------------

    distances = (
        (
            cluster_vectors_dense
            - centroid
        ) ** 2
    ).sum(
        axis=1
    )


    # --------------------------------------------------------
    # 11.6 Sort examples by distance
    # --------------------------------------------------------

    # The smallest distances correspond to the most
    # representative messages.
    closest_order = np.argsort(
        distances
    )


    # Select up to 10 representative examples.
    selected_positions = closest_order[
        :EXAMPLES_PER_CLUSTER
    ]


    # --------------------------------------------------------
    # 11.7 Display examples
    # --------------------------------------------------------

    for rank, local_position in enumerate(
        selected_positions,
        start=1
    ):

        # Convert the local cluster position back into the
        # original DataFrame position.
        original_position = (
            cluster_positions[local_position]
        )


        # Retrieve the actual customer message.
        message = df.loc[
            original_position,
            "customer_message"
        ]


        # Retrieve the conversation ID where available.
        if "conversation_id" in df.columns:

            conversation_id = df.loc[
                original_position,
                "conversation_id"
            ]

        else:

            conversation_id = ""


        print(
            f"\n{rank}. {message}"
        )


        # Save the representative example.
        representative_rows.append(
            {
                "cluster":
                    cluster_id,

                "rank":
                    rank,

                "conversation_id":
                    conversation_id,

                "customer_message":
                    message,
            }
        )


# ============================================================
# 12. SAVE REPRESENTATIVE EXAMPLES
# ============================================================

representative_df = pd.DataFrame(
    representative_rows
)


representative_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("INTENT DISCOVERY COMPLETE")
print("=" * 70)

print(
    f"\nNumber of exploratory clusters: "
    f"{NUMBER_OF_CLUSTERS}"
)

print(
    f"Examples per cluster: "
    f"{EXAMPLES_PER_CLUSTER}"
)

print(
    f"\nRepresentative examples saved to:"
)

print(
    OUTPUT_PATH
)

print(
    f"\nOutput exists:"
    f" {OUTPUT_PATH.exists()}"
)