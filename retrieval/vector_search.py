import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.reranker import rerank



CHROMA_PATH = (
    "data/embeddings/chroma_db"
)

COLLECTION_NAME = (
    "apple_10k"
)



def load_database():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    collection = client.get_collection(
        COLLECTION_NAME
    )


    return collection



def retrieve_vector(
        question,
        collection,
        model,
        top_k=5
):

    embedding = model.encode(
        question
    ).tolist()


    results = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=top_k

    )


    documents = results["documents"][0]

    metadata = results["metadatas"][0]


    combined=[]


    for doc, meta in zip(
        documents,
        metadata
    ):

        combined.append(
            {
                "text": doc,
                "metadata": meta
            }
        )


    return combined

def vector_search(query, top_k=5):

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    collection = client.get_collection(
        COLLECTION_NAME
    )


    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    embedding = model.encode(
        query
    ).tolist()


    results = collection.query(
        query_embeddings=[
            embedding
        ],
        n_results=top_k
    )


    output=[]


    for i in range(top_k):

        output.append(
            {
                "text":
                results["documents"][0][i],

                "metadata":
                results["metadatas"][0][i],

                "score":
                results["distances"][0][i]
            }
        )


    return output

def main():

    print(
        "Loading database..."
    )


    collection = load_database()


    print(
        "Loading embedding model..."
    )


    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    question = input(
        "Ask a question: "
    )


    results = retrieve_vector(
        question,
        collection,
        model
    )


    print("\nInitial Vector Results")
    print("=" * 60)


    for item in results:

        print(
            item["text"][:300]
        )

        print("-"*60)



    print(
        "\nRunning reranker..."
    )


    reranked = rerank(
        question,
        results
    )


    print(
        "\nRERANKED RESULTS"
    )

    print("=" * 60)



    for item in reranked:


        print(
            "Score:",
            item["score"]
        )


        print(
            item["text"][:800]
        )


        print(
            item["metadata"]
        )


        print("-"*60)




if __name__ == "__main__":

    main()