from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path(
    "data/raw/10-Q4-2024-As-Filed.pdf"
)

OUTPUT_PATH = Path(
    "data/processed/apple_report.txt"
)

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page_number, page in enumerate(reader.pages):
        print(f"Processing page {page_number + 1}")

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def main():

    text = extract_text(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8"
    )

    print(
        "Extraction complete!"
    )

    print(
        f"Characters extracted: {len(text)}"
    )


if __name__ == "__main__":
    main()