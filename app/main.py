import sys
import os
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st

from retrieval.pipeline import ask_question
from monitoring.logger import log_interaction


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "Financial RAG Assistant"
)

st.write(
    """
Ask questions about company financial reports.
The assistant uses hybrid retrieval, reranking,
and a language model to answer from financial filings.
"""
)


# ==========================================================
# QUESTION INPUT
# ==========================================================

question = st.text_input(
    "Enter your question:"
)


if question:

    st.subheader(
        "Question"
    )

    st.write(
        question
    )


    # ======================================================
    # RESPONSE TIMER
    # ======================================================

    start_time = time.time()


    with st.spinner(
        "Searching financial reports..."
    ):

        result = ask_question(
            question
        )


    response_time = round(
        time.time() - start_time,
        2
    )


    # ======================================================
    # ANSWER
    # ======================================================

    answer = result["answer"]


    st.subheader(
        "Answer"
    )


    st.write(
        answer
    )


    st.caption(
        f"Response time: {response_time} seconds"
    )


    # ======================================================
    # SOURCES
    # ======================================================

    st.subheader(
        "Retrieved Sources"
    )


    sources = result.get(
        "sources",
        []
    )


    for i, source in enumerate(
        sources
    ):

        with st.expander(
            f"Source {i + 1}"
        ):

            st.write(
                source["text"]
            )


            if "metadata" in source:

                st.write(
                    "Metadata"
                )

                st.json(
                    source["metadata"]
                )


    # ======================================================
    # FEEDBACK
    # ======================================================

    st.subheader(
        "Was this answer helpful?"
    )


    if "feedback_submitted" not in st.session_state:

        st.session_state.feedback_submitted = False


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "👍 Helpful"
        ):

            log_interaction(

                question=question,

                answer=answer,

                feedback="positive",

                response_time=response_time

            )


            st.session_state.feedback_submitted = True



    with col2:

        if st.button(
            "👎 Not Helpful"
        ):

            log_interaction(

                question=question,

                answer=answer,

                feedback="negative",

                response_time=response_time

            )


            st.session_state.feedback_submitted = True



    if st.session_state.feedback_submitted:

        st.success(
            "Feedback recorded."
        )