# ============================================================
# Simple Gemini API connectivity test
# ============================================================

from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai


# Find project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Load .env.
load_dotenv(
    PROJECT_ROOT / ".env"
)


# Read API key.
api_key = os.getenv(
    "GEMINI_API_KEY"
)


if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found."
    )


# Create Gemini client.
client = genai.Client(
    api_key=api_key
)


# Test the current model.
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=(
        "Classify this customer message in one sentence: "
        "Where is my Amazon package?"
    )
)


print("=" * 60)
print("GEMINI TEST SUCCESSFUL")
print("=" * 60)
print("\nResponse:")
print(response.text)