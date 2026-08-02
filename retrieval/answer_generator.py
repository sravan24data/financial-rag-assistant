from transformers import pipeline


_generator = None


def load_generator():

    global _generator

    if _generator is None:

        _generator = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            do_sample=False
        )

    return _generator



def generate_answer(query, chunks):

    model = load_generator()


    # Use only top reranked chunks
    selected_chunks = chunks[:2]


    # ======================================================
    # DOMAIN FILTERING
    # Keep cybersecurity questions focused on cyber risks
    # ======================================================

    query_lower = query.lower()


    if (
        "cybersecurity" in query_lower
        or "cyber security" in query_lower
    ):

        cyber_terms = [

            "cybersecurity",
            "malicious attacks",
            "unauthorized access",
            "ransomware",
            "confidential information",
            "security incidents",
            "information security"

        ]


        selected_chunks = [

            c

            for c in selected_chunks

            if any(

                term in c["text"].lower()

                for term in cyber_terms

            )

        ]


        # Fallback if filtering removes all chunks

        if not selected_chunks:

            selected_chunks = chunks[:2]



    context_parts = []


    for c in selected_chunks:

        metadata = c.get(
            "metadata",
            {}
        )


        source = metadata.get(
            "source",
            "unknown"
        )


        company = metadata.get(
            "company",
            "unknown"
        )


        document_type = metadata.get(
            "document_type",
            "unknown"
        )


        context_parts.append(

            f"""
Source: {source}
Company: {company}
Document: {document_type}

Content:
{c['text']}
"""

        )


    context = "\n\n".join(context_parts)



    prompt = f"""
You answer questions about SEC filings.

Rules:
- Use ONLY the provided context.
- Do not use outside knowledge.
- Do not make assumptions.
- Do not add information that is not directly supported by the filing.
- If information is missing, say:
  "The information is not available in the provided filing."


For cybersecurity risk questions:

- Include only cybersecurity-related risks disclosed in the filing.
- Focus on threats, attacks, unauthorized access, ransomware,
  confidential information exposure, security incidents,
  and disruption of systems or operations.
- Exclude unrelated business risks such as:
  currency fluctuations,
  competition,
  macroeconomic conditions,
  geopolitical events,
  and general business interruptions.
- Exclude cybersecurity governance information unless asked.
- Do not mention:
  security teams,
  committees,
  oversight,
  enterprise risk management,
  or mitigation activities.
- Do not combine unrelated sections of the filing.
- Do not add introductory or concluding commentary.


Context:

{context}


Question:

{query}


Answer:
"""


    response = model(

        prompt,

        max_new_tokens=250,

        do_sample=False

    )


    answer = response[0]["generated_text"]


    # Remove prompt if model repeats it

    if "Answer:" in answer:

        answer = answer.split(

            "Answer:"

        )[-1].strip()


    return answer