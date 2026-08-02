import json


FILE = "data/processed/apple_chunks.json"


with open(FILE, encoding="utf-8") as f:
    chunks = json.load(f)


for chunk in chunks:

    if "One Apple Park Way" in chunk["text"]:

        print("FOUND CHUNK")
        print("=" * 50)
        print(chunk["text"])
        print()
        print(chunk["metadata"])