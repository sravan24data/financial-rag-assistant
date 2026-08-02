import csv
import os
from datetime import datetime


LOG_FILE = "monitoring/interactions.csv"



def log_interaction(
    question,
    answer,
    feedback,
    response_time
):


    # ======================================================
    # NORMALIZE FEEDBACK VALUES
    # ======================================================

    if feedback in [
        "👍 Yes",
        "👍",
        "positive"
    ]:
        feedback = "positive"


    elif feedback in [
        "👎 No",
        "👎",
        "negative"
    ]:
        feedback = "negative"



    # ======================================================
    # ENSURE DIRECTORY EXISTS
    # ======================================================

    os.makedirs(
        "monitoring",
        exist_ok=True
    )



    file_exists = os.path.isfile(
        LOG_FILE
    )



    # ======================================================
    # WRITE LOG ENTRY
    # ======================================================

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.writer(
            f
        )



        if not file_exists:

            writer.writerow(
                [
                    "timestamp",
                    "question",
                    "answer",
                    "feedback",
                    "response_time"
                ]
            )



        writer.writerow(
            [
                datetime.now().isoformat(),

                question,

                answer,

                feedback,

                response_time
            ]
        )