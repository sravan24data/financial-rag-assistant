import os
import pandas as pd
import streamlit as st


LOG_FILE = "monitoring/interactions.csv"


st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📈",
    layout="wide"
)



st.title(
    "RAG Monitoring Dashboard"
)


st.write(
    """
Monitor user questions, feedback,
and system performance metrics.
"""
)



# ==========================
# LOAD DATA
# ==========================

if not os.path.exists(LOG_FILE):

    st.info(
        "No interactions have been logged yet."
    )

    st.stop()



df = pd.read_csv(
    LOG_FILE
)



if df.empty:

    st.info(
        "No data available."
    )

    st.stop()



# ==========================
# DATA CLEANING
# ==========================


df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)



# Handle old logs

if "response_time" not in df.columns:

    df["response_time"] = 0



df["response_time"] = pd.to_numeric(
    df["response_time"],
    errors="coerce"
)



df["question_length"] = (

    df["question"]

    .astype(str)

    .apply(len)

)



# ==========================
# METRICS
# ==========================


st.subheader(
    "System Overview"
)


col1,col2,col3,col4,col5 = st.columns(5)



with col1:

    st.metric(
        "Questions",
        len(df)
    )



with col2:

    positive = (

        df["feedback"]

        .isin(
            [
                "positive",
                "👍 Yes"
            ]
        )

        .sum()

    )


    st.metric(
        "Positive",
        positive
    )



with col3:

    negative = (

        df["feedback"]

        .isin(
            [
                "negative",
                "👎 No"
            ]
        )

        .sum()

    )


    st.metric(
        "Negative",
        negative
    )



with col4:

    st.metric(
        "Avg Response",
        round(
            df["response_time"].mean(),
            2
        )
    )



with col5:

    satisfaction = round(

        positive / len(df) * 100,

        2

    )


    st.metric(
        "Satisfaction %",
        f"{satisfaction}%"
    )



# ==========================
# CHART 1
# ==========================

st.subheader(
    "1. Feedback Distribution"
)


st.bar_chart(
    df["feedback"].value_counts()
)



# ==========================
# CHART 2
# ==========================


st.subheader(
    "2. Questions Over Time"
)


daily_questions = (

    df

    .set_index(
        "timestamp"
    )

    .resample(
        "D"
    )

    .size()

)


st.line_chart(
    daily_questions
)



# ==========================
# CHART 3
# ==========================


st.subheader(
    "3. Response Time Trend"
)


daily_response = (

    df

    .set_index(
        "timestamp"
    )

    ["response_time"]

    .resample(
        "D"
    )

    .mean()

)


st.line_chart(
    daily_response
)



# ==========================
# CHART 4
# ==========================


st.subheader(
    "4. Response Time Distribution"
)


st.bar_chart(
    df["response_time"]
)



# ==========================
# CHART 5
# ==========================


st.subheader(
    "5. Question Length Distribution"
)


st.bar_chart(
    df["question_length"]
)



# ==========================
# EXTRA CHART
# ==========================


st.subheader(
    "6. Feedback Trend Over Time"
)


feedback_trend = (

    df

    .set_index(
        "timestamp"
    )

    ["feedback"]

    .resample(
        "D"
    )

    .count()

)


st.line_chart(
    feedback_trend
)



# ==========================
# TABLE
# ==========================


st.subheader(
    "Recent Interactions"
)



st.dataframe(

    df[
        [
            "timestamp",
            "question",
            "feedback",
            "response_time"
        ]
    ]

    .sort_values(
        "timestamp",
        ascending=False
    )

)