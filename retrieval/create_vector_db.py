import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path(
    "data/processed/apple_chunks.json"
)

CHROMA_PATH = (
    "data/embeddings/chroma_db"
)


COLLECTION_NAME = "apple_10k"


def load_chunks():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks"
    )


    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    print("Creating Chroma database...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    # Remove old collection if it exists
    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except:
        pass


    collection = client.create_collection(
        name=COLLECTION_NAME
    )


    documents = []
    embeddings = []
    ids = []
    metadatas = []


    print("Creating embeddings...")


    for chunk in chunks:

        embedding = model.encode(
            chunk["text"]
        ).tolist()


        ids.append(
            str(chunk["id"])
        )

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            embedding
        )

        metadatas.append(
            chunk["metadata"]
        )


    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


    print(
        "Vector database created!"
    )

    print(
        f"Stored {collection.count()} documents"
    )


if __name__ == "__main__":
    main()