from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = Path(
    "data/processed/apple_report.txt"
)

OUTPUT_FILE = Path(
    "data/processed/apple_chunks.json"
)


def main():

    print("Reading document...")

    text = INPUT_FILE.read_text(
        encoding="utf-8"
    )


    print("Splitting document...")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


    chunks = splitter.split_text(text)


    output = []


    for index, chunk in enumerate(chunks):

        output.append(
            {
                "id": index,
                "text": chunk,
                "metadata": {
                    "company": "Apple",
                    "document_type": "10-K",
                    "source": "Apple-2024-10-K-As-Filed.pdf",
                    "chunk_number": index
                }
            }
        )


    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        f"Created {len(output)} chunks"
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()