import json
from rank_bm25 import BM25Okapi


DATA_PATH = "data/processed/apple_chunks.json"


def load_chunks():

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def tokenize(text):

    return text.lower().split()


def build_bm25(chunks):

    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    return bm25


def search(query, chunks, bm25, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]

def bm25_search(query, top_k=5):

    chunks = load_chunks()

    corpus = [
        c["text"]
        for c in chunks
    ]

    tokenized = [
        doc.lower().split()
        for doc in corpus
    ]


    bm25 = BM25Okapi(tokenized)


    scores = bm25.get_scores(
        query.lower().split()
    )


    ranked = sorted(
        zip(chunks, scores),
        key=lambda x:x[1],
        reverse=True
    )


    results=[]


    for chunk, score in ranked[:top_k]:

        results.append(
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score
            }
        )


    return results

def main():

    chunks = load_chunks()

    bm25 = build_bm25(chunks)


    question = input(
        "Ask a question: "
    )


    results = search(
        question,
        chunks,
        bm25
    )


    print("\nBM25 RESULTS")
    print("=" * 60)


    for chunk, score in results:

        print(
            "Score:",
            score
        )

        print(
            chunk["text"][:500]
        )

        print(
            chunk["metadata"]
        )

        print("-" * 60)



if __name__ == "__main__":

    main()