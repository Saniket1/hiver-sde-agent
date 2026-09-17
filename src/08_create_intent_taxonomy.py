# ============================================================
# Hiver SDE Intern Assignment
# Step 12 - Create Final Intent Taxonomy
# ============================================================
#
# PURPOSE:
# Store the final AmazonHelp intent taxonomy that was derived
# from the 500-message discovery sample and reviewed manually.
#
# IMPORTANT:
# These are now the frozen intent labels that will be used by:
#
#   1. Golden-set labeling
#   2. Intent classifier
#   3. Evaluation
#   4. Confusion analysis
#
# We should not casually change these labels after evaluation
# starts because that would make comparison difficult.
# ============================================================

from pathlib import Path
import json


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

# Locate the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Save the final taxonomy in the processed data directory.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "intents.json"
)


# Make sure the output directory exists.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. FINAL TAXONOMY
# ------------------------------------------------------------

intents = [
    {
        "name": "Order_Delivery_Tracking",
        "definition": (
            "Customer wants to know where an order or package is, "
            "its tracking status, or its expected arrival status."
        ),
        "confusable_with": [
            "Delivery_Delay_or_Missed_Date"
        ]
    },
    {
        "name": "Delivery_Delay_or_Missed_Date",
        "definition": (
            "Customer reports that an order is late, delayed, "
            "or has missed its promised or expected delivery date."
        ),
        "confusable_with": [
            "Order_Delivery_Tracking"
        ]
    },
    {
        "name": "Order_Cancellation_or_Change",
        "definition": (
            "Customer wants to cancel or change an order, "
            "or reports that an order was unexpectedly cancelled."
        ),
        "confusable_with": [
            "Returns_Refunds_or_Replacements"
        ]
    },
    {
        "name": "Missing_Wrong_or_Damaged_Item",
        "definition": (
            "Customer reports that an item is missing, wrong, "
            "incomplete, damaged, or that an expected package "
            "contains an incorrect item."
        ),
        "confusable_with": [
            "Returns_Refunds_or_Replacements",
            "Delivery_Delay_or_Missed_Date"
        ]
    },
    {
        "name": "Returns_Refunds_or_Replacements",
        "definition": (
            "Customer asks about returning an item, receiving "
            "a refund, or obtaining a replacement."
        ),
        "confusable_with": [
            "Missing_Wrong_or_Damaged_Item",
            "Payment_or_Charge_Issue"
        ]
    },
    {
        "name": "Payment_or_Charge_Issue",
        "definition": (
            "Customer reports a payment problem, unexpected charge, "
            "cashback issue, billing issue, or other payment-related problem."
        ),
        "confusable_with": [
            "Returns_Refunds_or_Replacements",
            "Account_Login_or_Security"
        ]
    },
    {
        "name": "Account_Login_or_Security",
        "definition": (
            "Customer has an account access, login, verification, "
            "authentication, or security/unauthorized-activity problem."
        ),
        "confusable_with": [
            "Payment_or_Charge_Issue"
        ]
    },
    {
        "name": "Product_Availability_or_Pricing",
        "definition": (
            "Customer asks whether a product is available, "
            "when it will be in stock, about its price, "
            "promotions, discounts, or sales."
        ),
        "confusable_with": [
            "Product_or_Service_Information"
        ]
    },
    {
        "name": "Product_or_Service_Information",
        "definition": (
            "Customer requests general information about an Amazon "
            "product or service, including compatibility or how a "
            "service/feature works."
        ),
        "confusable_with": [
            "Product_Availability_or_Pricing",
            "Prime_Video_Content_Availability"
        ]
    },
    {
        "name": "Prime_Video_Content_Availability",
        "definition": (
            "Customer asks whether movies, shows, seasons, or other "
            "Prime Video content are available, when they will be "
            "released, whether they are available in a region, or "
            "requests additional content."
        ),
        "confusable_with": [
            "Product_or_Service_Information"
        ]
    }
]


# ------------------------------------------------------------
# 3. CREATE OUTPUT OBJECT
# ------------------------------------------------------------

taxonomy = {
    "brand": "AmazonHelp",
    "version": "1.0",
    "intent_count": len(intents),
    "intents": intents
}


# ------------------------------------------------------------
# 4. SAVE JSON
# ------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        taxonomy,
        file,
        indent=2,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 5. VERIFY
# ------------------------------------------------------------

print("=" * 70)
print("AMAZONHELP INTENT TAXONOMY CREATED")
print("=" * 70)

print(
    f"\nBrand: {taxonomy['brand']}"
)

print(
    f"Number of intents: "
    f"{taxonomy['intent_count']}"
)

print("\nIntent labels:")

for intent in intents:
    print(
        f"- {intent['name']}"
    )

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print(
    f"\nOutput exists: "
    f"{OUTPUT_PATH.exists()}"
)