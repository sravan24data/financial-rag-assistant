from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query, results, top_k=20):

    print("Loading reranker...")

    model = CrossEncoder(
        MODEL_NAME
    )


    # Prepare query-document pairs
    pairs = []

    for item in results:

        pairs.append(
            (
                query,
                item["text"]
            )
        )


    # Get reranking scores
    scores = model.predict(
        pairs
    )


    # Attach reranking scores
    reranked = []

    for item, score in zip(results, scores):

        reranked.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "rerank_score": float(score)
            }
        )


    # Sort highest score first
    reranked = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    return reranked[:top_k]