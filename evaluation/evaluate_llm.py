import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from transformers import pipeline

from retrieval.pipeline import ask_question
from retrieval.hybrid_search import hybrid_search, load_chunks


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    do_sample=False
)


TEST_CASES = [
    {
        "question": "What are Apple's cybersecurity risks?",
        "keywords": [
            "cybersecurity",
            "malicious attacks",
            "unauthorized access",
            "confidential information"
        ]
    },
    {
        "question": "What was Apple's total net sales for fiscal year 2024?",
        "keywords": [
            "net sales",
            "2024",
            "revenue"
        ]
    }
]


def keyword_score(answer, keywords):

    answer = answer.lower()

    matches = 0

    for keyword in keywords:

        if keyword.lower() in answer:
            matches += 1

    return matches / len(keywords)



def generate_baseline_answer(question, context):

    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}

Answer:
"""


    response = generator(
        prompt,
        max_new_tokens=250,
        do_sample=False
    )


    return response[0]["generated_text"].split(
        "Answer:"
    )[-1].strip()



def build_context(chunks):

    return "\n\n".join(
        [
            chunk["text"]
            for chunk in chunks[:5]
        ]
    )



def evaluate():

    chunks = load_chunks()


    baseline_total = 0
    production_total = 0


    for case in TEST_CASES:

        question = case["question"]

        keywords = case["keywords"]


        print("\nQUESTION")
        print(question)


        retrieved = hybrid_search(
            question,
            chunks,
            top_k=5
        )


        context = build_context(
            retrieved
        )


        # -----------------------------
        # Baseline Prompt
        # -----------------------------

        baseline_answer = generate_baseline_answer(
            question,
            context
        )


        baseline_score = keyword_score(
            baseline_answer,
            keywords
        )


        baseline_total += baseline_score


        print("\nBaseline Answer:")
        print(baseline_answer)

        print(
            "Baseline Score:",
            baseline_score
        )


        # -----------------------------
        # Production Pipeline
        # -----------------------------

        production_result = ask_question(
            question
        )


        production_answer = production_result["answer"]


        production_score = keyword_score(
            production_answer,
            keywords
        )


        production_total += production_score


        print("\nProduction Answer:")
        print(production_answer)


        print(
            "Production Score:",
            production_score
        )


    print("\nFINAL RESULTS")
    print("====================")


    print(
        "Baseline Average:",
        baseline_total / len(TEST_CASES)
    )


    print(
        "Production Average:",
        production_total / len(TEST_CASES)
    )


    if production_total >= baseline_total:

        print(
            "\nSelected Approach: Production Prompt"
        )

    else:

        print(
            "\nSelected Approach: Baseline Prompt"
        )



if __name__ == "__main__":

    evaluate()