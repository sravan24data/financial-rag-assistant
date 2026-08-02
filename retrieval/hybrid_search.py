import json
import re
import os

import chromadb
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval.reranker import rerank

# ==========================================================
# CONFIGURATION
# ==========================================================

CHROMA_PATH = "data/embeddings/chroma_db"

COLLECTION_NAME = "apple_10k"

CHUNKS_FILE = "data/processed/apple_chunks.json"

EMBED_MODEL = "all-MiniLM-L6-v2"

HF_TOKEN = os.getenv("HF_TOKEN")


_embedding_model = None



# ==========================================================
# LOAD DATA
# ==========================================================

def load_chunks():

    with open(
        CHUNKS_FILE,
        encoding="utf-8"
    ) as f:

        return json.load(f)



# ==========================================================
# TEXT HELPERS
# ==========================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )



def clean_compare(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()



# ==========================================================
# BM25 SEARCH
# ==========================================================

def bm25_search(
        chunks,
        query,
        top_k=30
):

    documents = [

        c["text"]

        for c in chunks

    ]


    tokenized_docs = [

        tokenize(doc)

        for doc in documents

    ]


    bm25 = BM25Okapi(
        tokenized_docs
    )


    scores = bm25.get_scores(
        tokenize(query)
    )


    ranked = sorted(

        enumerate(scores),

        key=lambda x: x[1],

        reverse=True

    )


    return ranked[:top_k]



# ==========================================================
# EMBEDDING MODEL CACHE
# ==========================================================

def get_embedding_model():

    global _embedding_model


    if _embedding_model is None:

        _embedding_model = SentenceTransformer(
            EMBED_MODEL
        )


    return _embedding_model



# ==========================================================
# VECTOR SEARCH
# ==========================================================

def vector_search(
        query,
        top_k=30
):

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    collection = client.get_collection(
        COLLECTION_NAME
    )


    model = get_embedding_model()


    embedding = model.encode(
        query
    ).tolist()


    result = collection.query(

        query_embeddings=[

            embedding

        ],

        n_results=top_k

    )


    output = []


    for i, text in enumerate(

        result["documents"][0]

    ):

        output.append(

            {

                "text": text,


                "metadata":
                    result["metadatas"][0][i],


                "distance":
                    result["distances"][0][i]

            }

        )


    return output



# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize(values):

    values = np.array(

        values,

        dtype=float

    )


    if len(values) == 0:

        return values



    if values.max() == values.min():

        return np.ones_like(values)



    return (

        values - values.min()

    ) / (

        values.max() - values.min()

    )

# ==========================================================
# RISK RELEVANCE FILTER
# ==========================================================

def is_relevant_risk_chunk(
        text,
        query
):

    text = text.lower()

    query = query.lower()


    if "risk" not in query:

        return True



    risk_terms = [

        "item 1a. risk factors",

        "risk factors",

        "cybersecurity",

        "malicious attacks",

        "government investigations",

        "legal proceedings",

        "regulatory changes",

        "macroeconomic",

        "supplier",

        "supply chain",

        "competition",

        "privacy",

        "intellectual property"

    ]


    return any(

        term in text

        for term in risk_terms

    )



# ==========================================================
# TABLE OF CONTENTS FILTER
# ==========================================================

def is_table_of_contents(text):

    text = text.lower()


    toc_patterns = [

        "table of contents",

        "item 1. business",

        "item 1a. risk factors",

        "part i",

        "part ii",

        "page"

    ]


    matches = sum(

        1

        for pattern in toc_patterns

        if pattern in text

    )


    return matches >= 3




# ==========================================================
# HYBRID SEARCH
# ==========================================================

def hybrid_search(
        query,
        chunks,
        top_k=30
):

    combined = {}



    # ======================================================
    # BM25 SEARCH
    # ======================================================

    bm25_results = bm25_search(

        chunks,

        query

    )


    bm25_scores = normalize(

        [

            score

            for _, score in bm25_results

        ]

    )



    for ((idx, _), score) in zip(

        bm25_results,

        bm25_scores

    ):


        combined[idx] = {


            "text":

                chunks[idx]["text"],


            "metadata":

                chunks[idx].get(

                    "metadata",

                    chunks[idx]

                ),


            "score":

                float(score) * 0.5

        }



    # ======================================================
    # VECTOR SEARCH
    # ======================================================

    vector_results = vector_search(

        query

    )


    vector_scores = normalize(

        [

            max(

                0,

                1 - r["distance"]

            )

            for r in vector_results

        ]

    )



    for result, score in zip(

        vector_results,

        vector_scores

    ):


        match = None



        for item in combined.values():


            if clean_compare(

                item["text"][:200]

            ) == clean_compare(

                result["text"][:200]

            ):


                match = item

                break




        if match:


            match["score"] += (

                float(score) * 0.5

            )


        else:


            combined[len(combined)] = {


                "text":

                    result["text"],


                "metadata":

                    result["metadata"],


                "score":

                    float(score) * 0.5

            }



    results = list(

        combined.values()

    )



    # ======================================================
    # DOMAIN BOOSTING
    # ======================================================

    q = query.lower()



    for r in results:


        text = r["text"].lower()



        # --------------------------------------------------
        # Risk Questions
        # --------------------------------------------------
        

        risk_queries = [
            "risk",
            "risks",
            "risk factors",
            "uncertainty",
            "challenge",
            "threat"
            ]

        if any(term in q for term in risk_queries):
        
            # Boost only real Item 1A sections
            # Avoid table of contents

            if (

                "item 1a" in text
                
                and
                
                "risk factors" in text

                and

                "table of contents" not in text

            ):

                r["score"] += 60



            elif "risk factors" in text:

                r["score"] += 10




            risk_terms = [

                "cybersecurity",

                "malicious attacks",

                "government investigations",

                "legal proceedings",

                "regulatory changes",

                "macroeconomic",

                "supplier",

                "supply chain",

                "competition",

                "privacy",

                "intellectual property"

            ]



            for term in risk_terms:


                if term in text:


                    r["score"] += 3




            # Remove unrelated health section

            if "health and safety" in text:


                r["score"] -= 15




        # --------------------------------------------------
        # Headquarters Questions
        # --------------------------------------------------

        if (

            "headquarters" in q

            and

            "cupertino" in text

        ):

            r["score"] += 10




        # --------------------------------------------------
        # Revenue Questions
        # --------------------------------------------------

        if (

            "net sales" in q

            and

            "net sales" in text

        ):

            r["score"] += 10




        # --------------------------------------------------
        # Research and Development
        # --------------------------------------------------

        if (

            "research" in q

            and

            "development" in q

            and

            "research and development" in text

        ):

            r["score"] += 10




    # ======================================================
    # PRE-RERANK FILTERING
    # ======================================================

    if "risk" in q:


        results = [

            r

            for r in results

            if (

                is_relevant_risk_chunk(

                    r["text"],

                    query

                )

                and

                not is_table_of_contents(

                    r["text"]

                )

            )

        ]



    results.sort(

        key=lambda x: x["score"],

        reverse=True

    )



    return results[:top_k]

# ==========================================================
# FINAL RERANKING CORRECTIONS
# ==========================================================

def apply_rerank_corrections(
        reranked,
        query
):

    q = query.lower()


    for r in reranked:


        text = r["text"].lower()


        score = float(

            r.get(

                "score",

                r.get(

                    "rerank_score",

                    0

                )

            )

        )


        # ==================================================
        # Risk Question Corrections
        # ==================================================

        if "risk" in q:


            # ----------------------------------------------
            # True Item 1A Risk Factors Section
            # ----------------------------------------------

            if (

                "item 1a. risk factors" in text

                and

                "table of contents" not in text

            ):

                score += 70



            # Main risk introduction paragraph

            if (

                "company’s business, reputation"

                in text

            ):

                score += 40




            # ----------------------------------------------
            # Actual risk disclosure language
            # ----------------------------------------------

            real_risk_indicators = [

                "the company experiences",

                "can materially adversely affect",

                "may adversely affect",

                "could materially adversely affect",

                "risks include",

                "factors include"

            ]


            for phrase in real_risk_indicators:


                if phrase in text:

                    score += 15




            # ----------------------------------------------
            # Specific risk categories
            # ----------------------------------------------

            risk_sections = [

                "cybersecurity",

                "malicious attacks",

                "government investigations",

                "legal proceedings",

                "regulatory changes",

                "macroeconomic",

                "supplier",

                "supply chain",

                "competition",

                "privacy",

                "intellectual property"

            ]



            for term in risk_sections:


                if term in text:

                    score += 8




            # ----------------------------------------------
            # Penalize unrelated safety sections
            # ----------------------------------------------

            if "health and safety" in text:

                score -= 20




            # ----------------------------------------------
            # Penalize references only
            # Example:
            # "For a discussion of cybersecurity risks,
            #  see Item 1A..."
            # ----------------------------------------------

            reference_phrases = [

                "for a discussion of",

                "see item 1a of this form 10-k",

                "under the heading"

            ]



            for phrase in reference_phrases:


                if phrase in text:

                    score -= 80





        # ==================================================
        # Remove Filing Noise
        # ==================================================

        bad_sections = [

            "documents incorporated by reference",

            "registration statement",

            "exhibit 31.1",

            "i, timothy d. cook, certify",

            "i, luca maestri, certify",

            "principal accountant fees",

            "signature",

            "table of contents"

        ]



        for bad in bad_sections:


            if bad in text:

                score -= 100




        # Save corrected score

        r["final_score"] = score





    # Final ordering

    reranked.sort(

        key=lambda x: x["final_score"],

        reverse=True

    )


    return reranked

def filter_domain_chunks(results, query):

    q = query.lower()

    filtered = results


    if "cybersecurity" in q or "cyber security" in q:

        cyber_terms = [

            "cybersecurity",
            "malicious attacks",
            "unauthorized access",
            "ransomware",
            "confidential information",
            "systems",
            "security incidents",
            "information security"

        ]


        filtered = [

            r

            for r in results

            if any(

                term in r["text"].lower()

                for term in cyber_terms

            )

        ]


    return filtered


# ==========================================================
# MAIN
# ==========================================================

def main():


    chunks = load_chunks()



    query = input(

        "Ask a question: "

    )



    print(

        "\nRunning Hybrid Search"

    )

    print(

        "=" * 60

    )



    results = hybrid_search(

        query,

        chunks

    )



    print(

        "\nBefore Reranking"

    )

    print(

        "=" * 60

    )



    for r in results:


        print(

            "Hybrid Score:",

            r["score"]

        )


        print(

            r["text"][:400]

        )


        print(

            r["metadata"]

        )


        print(

            "-" * 60

        )




    print(

        "\nRunning Reranker"

    )

    print(

        "=" * 60

    )

    results = filter_domain_chunks(
        results,
        query
        )

    rerank_input = [

        {

            "text":

                r["text"],


            "metadata":

                r["metadata"]

        }

        for r in results

    ]



    reranked = rerank(

        query,

        rerank_input

    )



    reranked = apply_rerank_corrections(

        reranked,

        query

    )


    print(
        "\nReranking Complete"
        )

    print(
        "=" * 60
        )

    print("\nCHECK CHUNK")
    print(reranked[0])
    
    print("\nCHECK KEYS")
    print(reranked[0].keys())
    print(

        "\nRERANKED RESULTS"

    )

    print(

        "=" * 60

    )




    for r in reranked[:5]:


        print(

            "Final Score:",

            r["final_score"]

        )


        print(

            r["text"][:600]

        )


        print(

            r.get(

                "metadata",

                {}

            )

        )


        print(

            "-" * 60

        )

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()