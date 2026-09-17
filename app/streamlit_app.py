"""
Hiver SDE Intern Assignment
Streamlit demo for the AmazonHelp Support Agent.

This UI reuses the existing end-to-end agent implemented in:

src/20_support_agent.py

The existing agent already performs:
1. Intent classification
2. Historical FAISS retrieval
3. Escalation decision
4. Draft response generation

The Streamlit app only provides a user-friendly interface around it.
"""

from pathlib import Path
import importlib.util
import sys

import streamlit as st


# ============================================================
# 1. PROJECT PATH
# ============================================================

# The current file is:
#
# hiver-sde-agent/
# └── app/
#     └── streamlit_app.py
#
# parents[1] therefore points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. LOAD THE EXISTING SUPPORT AGENT
# ============================================================

# Your existing agent is named:
#
# src/20_support_agent.py
#
# Python cannot import a module beginning with a number using:
#
#     from src import support_agent
#
# Therefore, we load the file directly using importlib.
AGENT_FILE = PROJECT_ROOT / "src" / "20_support_agent.py"


# Check that the existing agent file actually exists.
if not AGENT_FILE.exists():

    st.error(
        f"Support-agent file was not found:\n{AGENT_FILE}"
    )

    st.stop()


# ------------------------------------------------------------
# Dynamically load 20_support_agent.py
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "amazonhelp_support_agent",
    AGENT_FILE,
)


# Stop with a useful message if Python cannot create
# the import specification.
if spec is None or spec.loader is None:

    st.error(
        "Could not load src/20_support_agent.py."
    )

    st.stop()


# Create a module object from the specification.
support_agent = importlib.util.module_from_spec(
    spec
)


# Register the dynamically loaded module.
sys.modules["amazonhelp_support_agent"] = support_agent


# Execute the module so that all models, functions,
# FAISS index, and configuration are initialized.
try:

    spec.loader.exec_module(
        support_agent
    )

except Exception as exc:

    st.error(
        "The existing support agent could not be loaded."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AmazonHelp AI Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# 4. PAGE HEADER
# ============================================================

st.title(
    "AmazonHelp AI Support Agent"
)

st.caption(
    "Hiver SDE Intern Take-Home — "
    "Intent Classification + Historical Retrieval + "
    "Escalation + Grounded Response"
)


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About this demo"
    )

    st.write(
        """
This Streamlit application is a lightweight interface
around the AmazonHelp support-agent pipeline.

The agent performs:

• Intent classification
• Historical support-case retrieval
• Escalation decision
• Escalation reason generation
• Grounded draft-response generation
"""
    )

    st.divider()

    st.info(
        "Prototype only. This demo is not connected to "
        "live Amazon order, account, refund, or ticketing APIs."
    )


# ============================================================
# 6. CUSTOMER MESSAGE INPUT
# ============================================================

customer_message = st.text_area(
    "Customer message",
    height=160,
    placeholder=(
        "Example: My package says delivered "
        "but I never received it."
    ),
)


# ============================================================
# 7. ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(
    "Analyze Customer Message",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 8. RUN THE SUPPORT AGENT
# ============================================================

if analyze_clicked:

    # Check that the customer entered a message.
    if not customer_message.strip():

        st.warning(
            "Please enter a customer message."
        )

        st.stop()


    # --------------------------------------------------------
    # Run the existing run_agent() function.
    # --------------------------------------------------------

    try:

        result = support_agent.run_agent(
            customer_message.strip()
        )

    except Exception as exc:

        st.error(
            "The support agent failed while processing "
            "the message."
        )

        st.exception(exc)

        st.stop()


    # ========================================================
    # 9. INTENT
    # ========================================================

    st.subheader(
        "1. Intent"
    )

    # Your existing agent returns:
    #
    # result["intent"]
    # result["intent_confidence"]

    predicted_intent = result.get(
        "intent",
        "Unavailable",
    )

    intent_confidence = result.get(
        "intent_confidence",
        0.0,
    )


    # Display intent and confidence side by side.
    intent_col, confidence_col = st.columns(
        2
    )


    with intent_col:

        st.metric(
            "Predicted Intent",
            predicted_intent,
        )


    with confidence_col:

        st.metric(
            "Intent Confidence",
            f"{float(intent_confidence):.2f}",
        )


    # ========================================================
    # 10. ESCALATION
    # ========================================================

    st.subheader(
        "2. Escalation Decision"
    )

    # Your existing agent returns:
    #
    # result["decision"]
    # result["escalation_reason"]

    decision = result.get(
        "decision",
        "UNKNOWN",
    )

    escalation_reason = result.get(
        "escalation_reason",
        "No reason provided.",
    )


    # Display the escalation state with an appropriate
    # Streamlit status component.
    if decision == "HUMAN":

        st.error(
            f"🔴 HUMAN ESCALATION\n\n"
            f"Reason: {escalation_reason}"
        )

    elif decision == "AUTO-HANDLE":

        st.success(
            f"🟢 AUTO-HANDLE\n\n"
            f"Reason: {escalation_reason}"
        )

    else:

        st.warning(
            f"Decision: {decision}\n\n"
            f"Reason: {escalation_reason}"
        )


    # ========================================================
    # 11. HISTORICAL SUPPORT EVIDENCE
    # ========================================================

    st.subheader(
        "3. Historical Support Evidence"
    )

    # Your existing agent returns the retrieved cases
    # under result["retrieved_cases"].
    retrieved_cases = result.get(
        "retrieved_cases",
        [],
    )


    # Show the retrieved historical cases.
    if retrieved_cases:

        for case in retrieved_cases:

            # Extract fields produced by the existing retriever.
            rank = case.get(
                "rank",
                "?",
            )

            similarity = case.get(
                "similarity",
                0.0,
            )

            historical_customer = case.get(
                "customer_message",
                "",
            )

            historical_response = case.get(
                "agent_response",
                "",
            )


            # Put every historical case inside an expandable
            # section to keep the interface readable.
            with st.expander(
                f"Historical Case {rank} "
                f"— Similarity {similarity:.4f}"
            ):

                st.markdown(
                    "**Historical customer message**"
                )

                st.write(
                    historical_customer
                )


                st.markdown(
                    "**Historical AmazonHelp response**"
                )

                st.write(
                    historical_response
                )


    else:

        st.info(
            "No historical support cases were retrieved."
        )


    # ========================================================
    # 12. DRAFT RESPONSE
    # ========================================================

    st.subheader(
        "4. Draft Response"
    )

    # Your existing agent returns:
    #
    # result["draft_response"]

    draft_response = result.get(
        "draft_response",
        "",
    )


    if draft_response:

        st.info(
            draft_response
        )

    else:

        st.warning(
            "No draft response was produced."
        )


    # ========================================================
    # 13. RESPONSE SOURCE
    # ========================================================

    st.subheader(
        "5. Response Source"
    )

    # This tells the user whether the response was generated
    # by Gemini or by the local fallback.
    response_source = result.get(
        "response_source",
        "UNKNOWN",
    )


    if response_source == "GEMINI":

        st.success(
            "Gemini-generated response"
        )

    elif response_source == "LOCAL_FALLBACK_QUOTA":

        st.warning(
            "Conservative local fallback — "
            "Gemini quota was unavailable."
        )

    elif response_source == "LOCAL_FALLBACK":

        st.warning(
            "Conservative local fallback."
        )

    elif response_source == "LOCAL_FALLBACK_ERROR":

        st.warning(
            "Conservative local fallback — "
            "Gemini returned an error."
        )

    else:

        st.info(
            response_source
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "AmazonHelp Support Agent — Hiver SDE Intern Take-Home"
)