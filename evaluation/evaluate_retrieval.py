import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import json

from retrieval.bm25_search import bm25_search
from retrieval.vector_search import vector_search
from retrieval.hybrid_search import hybrid_search, load_chunks


QUESTIONS_PATH = "evaluation/questions.json"


def keyword_match(text, keywords):

    text = text.lower()

    matches = 0

    for keyword in keywords:

        if keyword.lower() in text:
            matches += 1

    return matches / len(keywords)



def evaluate():

    with open(
        QUESTIONS_PATH,
        encoding="utf-8"
    ) as f:

        questions = json.load(f)


    chunks = load_chunks()


    scores = {
        "BM25": 0,
        "Vector": 0,
        "Hybrid": 0
    }


    for item in questions:

        question = item["question"]

        keywords = item["expected_keywords"]


        print("\nQUESTION")
        print(question)


        # ==========================================
        # BM25 SEARCH
        # ==========================================

        bm25_results = bm25_search(
            question,
            top_k=5
        )


        bm25_text = " ".join(
            [
                item["text"]
                for item in bm25_results
            ]
        )


        bm25_score = keyword_match(
            bm25_text,
            keywords
        )


        scores["BM25"] += bm25_score


        print(
            "BM25:",
            bm25_score
        )


        # ==========================================
        # VECTOR SEARCH
        # ==========================================

        vector_results = vector_search(
            question,
            top_k=5
        )


        vector_text = " ".join(
            [
                item["text"]
                for item in vector_results
            ]
        )


        vector_score = keyword_match(
            vector_text,
            keywords
        )


        scores["Vector"] += vector_score


        print(
            "Vector:",
            vector_score
        )


        # ==========================================
        # HYBRID SEARCH
        # ==========================================

        hybrid_results = hybrid_search(
            question,
            chunks,
            top_k=5
        )


        hybrid_text = " ".join(
            [
                item["text"]
                for item in hybrid_results
            ]
        )


        hybrid_score = keyword_match(
            hybrid_text,
            keywords
        )


        scores["Hybrid"] += hybrid_score


        print(
            "Hybrid:",
            hybrid_score
        )


    # ==========================================
    # FINAL RESULTS
    # ==========================================

    print("\nFINAL RESULTS")
    print("========================")


    averages = {}


    for method, score in scores.items():

        averages[method] = (
            score / len(questions)
        )

        print(
            f"{method} Average:",
            averages[method]
        )


    # ==========================================
    # SELECT BEST METHOD
    # Prefer Hybrid when tied
    # ==========================================

    best_score = max(
        averages.values()
    )


    best_methods = [
        method
        for method, score in averages.items()
        if score == best_score
    ]


    if "Hybrid" in best_methods:

        best_method = "Hybrid"

    else:

        best_method = best_methods[0]


    print(
        "\nBest Retrieval Method:",
        best_method
    )



if __name__ == "__main__":

    evaluate()