from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from prefect import flow, task

from ingestion.extract_text import (
    extract_text,
    PDF_PATH,
    OUTPUT_PATH
)

from ingestion import chunk_text
from retrieval import create_vector_db


@task(
    name="Extract PDF text"
)
def extract_pdf_task():

    print("Extracting PDF...")

    text = extract_text(
        PDF_PATH
    )

    OUTPUT_PATH.parent.mkdir(
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


@task(
    name="Create chunks"
)
def create_chunks_task():

    print("Creating chunks...")

    chunk_text.main()

    print(
        "Chunks created"
    )


@task(
    name="Create ChromaDB"
)
def create_vector_db_task():

    print("Creating vector database...")

    create_vector_db.main()

    print(
        "Vector database created"
    )


@flow(
    name="financial-rag-ingestion"
)
def ingestion_flow():

    extract_pdf_task()

    create_chunks_task()

    create_vector_db_task()


if __name__ == "__main__":
    ingestion_flow()