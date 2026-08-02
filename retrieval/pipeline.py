from retrieval.hybrid_search import hybrid_search, load_chunks
from retrieval.reranker import rerank
from retrieval.answer_generator import generate_answer
from retrieval.query_rewrite import rewrite_query



def ask_question(query):


    # ======================================================
    # QUERY REWRITING
    # ======================================================

    rewritten_query = rewrite_query(
        query
    )


    # ======================================================
    # LOAD KNOWLEDGE BASE
    # ======================================================

    chunks = load_chunks()



    # ======================================================
    # HYBRID RETRIEVAL
    # ======================================================

    results = hybrid_search(
        rewritten_query,
        chunks
    )



    # ======================================================
    # DOCUMENT RERANKING
    # ======================================================

    reranked = rerank(
        rewritten_query,
        results
    )



    # ======================================================
    # ANSWER GENERATION
    # ======================================================

    answer = generate_answer(
        query,
        reranked[:5]
    )



    return {
        "answer": answer,
        "sources": reranked[:5],
        "rewritten_query": rewritten_query
    }