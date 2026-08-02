# Financial RAG Assistant

A Retrieval-Augmented Generation (RAG) application for answering questions about SEC financial filings using hybrid retrieval, document reranking, and a local Large Language Model (LLM).

The application retrieves relevant sections from company filings and generates grounded answers using only the retrieved documents.

---

# Project Objective

Financial reports are lengthy and difficult to search manually.

This project builds a RAG system that allows users to ask natural language questions about SEC filings while ensuring that responses are grounded in the original filing.

Example questions:

- What were Apple's total net sales in 2024?
- What cybersecurity risks does Apple disclose?
- What are Apple's operating segments?
- What supply chain risks are discussed?

---

# Architecture

```
                    User Question
                          │
                          ▼
                  Query Rewriting
                          │
                          ▼
      ┌────────────────────────────┐
      │ Hybrid Retrieval           │
      │                            │
      │ • BM25 Search              │
      │ • Chroma Vector Search     │
      └────────────────────────────┘
                          │
                          ▼
              Cross Encoder Reranker
                          │
                          ▼
             Top Ranked Document Chunks
                          │
                          ▼
             Qwen2.5-1.5B-Instruct LLM
                          │
                          ▼
                   Generated Answer
                          │
                          ▼
          Streamlit UI + Monitoring Dashboard
```

---

# Features

- Hybrid retrieval (BM25 + Vector Search)
- Query rewriting
- Cross-Encoder document reranking
- Local LLM generation using Hugging Face Transformers
- Streamlit interface
- Monitoring dashboard
- User feedback collection
- Response time tracking
- Docker support
- Retrieval evaluation
- LLM evaluation

---

# Repository Structure

```
financial-rag-assistant/
│
├── app/
│   ├── main.py
│   └── pages/
│       └── 1_Monitoring.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── ingestion/
│   ├── extract_text.py
│   └── chunk_text.py
│
├── retrieval/
│   ├── hybrid_search.py
│   ├── vector_search.py
│   ├── bm25_search.py
│   ├── reranker.py
│   ├── query_rewrite.py
│   ├── answer_generator.py
│   └── pipeline.py
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_llm.py
│   └── questions.json
│
├── monitoring/
│   ├── logger.py
│   └── interactions.csv
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Dataset

The project uses Apple's SEC Form 10-K filing.

```
data/raw/10-Q4-2024-As-Filed.pdf
```

The PDF is converted into text, chunked, embedded, and indexed in ChromaDB.

---

# Retrieval Pipeline

The retrieval workflow is:

1. User enters a question.
2. Query rewriting expands important financial keywords.
3. BM25 retrieves lexical matches.
4. ChromaDB retrieves semantic matches.
5. Results are merged using hybrid search.
6. CrossEncoder reranks retrieved chunks.
7. Top chunks are passed to the LLM.
8. The answer is generated using only retrieved context.

---

# Query Rewriting

Example:

```
User:

What are Apple's risks?

↓

Expanded Query:

What are Apple's risks?
risk factors business risks threats affecting Apple operations
```

This improves recall during retrieval.

---

# Hybrid Retrieval

The project combines:

- BM25 keyword search
- SentenceTransformer embeddings
- ChromaDB vector similarity

Hybrid retrieval improves both precision and recall compared to either method alone.

---

# Document Reranking

Retrieved chunks are reranked using:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Only the highest scoring chunks are sent to the LLM.

---

# LLM

Model:

```
Qwen/Qwen2.5-1.5B-Instruct
```

Prompt rules ensure:

- only retrieved context is used
- no hallucinations
- no external knowledge
- cybersecurity questions remain domain specific

---

# Evaluation

## Retrieval Evaluation

Implemented in:

```
evaluation/evaluate_retrieval.py
```

Approaches evaluated:

- Vector Search
- BM25
- Hybrid Search
- Hybrid + Reranker

The best-performing retrieval pipeline is used in production.

---

## LLM Evaluation

Implemented in:

```
evaluation/evaluate_llm.py
```

Two prompting strategies are compared:

- Baseline Prompt
- Production Prompt

Example results:

| Prompt | Average Score |
|---------|---------------|
| Baseline | 0.125 |
| Production | 1.00 |

The production prompt is selected.

---

# Monitoring Dashboard

A Streamlit dashboard tracks system performance.

Collected metrics include:

- Total questions
- Positive feedback
- Negative feedback
- Average response time

Charts include:

- Feedback distribution
- Questions over time
- Response time trend
- Response time distribution
- Question length distribution

User interactions are stored in:

```
monitoring/interactions.csv
```

---

# Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/main.py
```

Monitoring dashboard:

```
http://localhost:8501
```

Select:

```
Monitoring
```

from the Streamlit sidebar.

---

# Running Evaluations

Retrieval evaluation:

```bash
python evaluation/evaluate_retrieval.py
```

LLM evaluation:

```bash
python evaluation/evaluate_llm.py
```

---

# Docker

Build:

```bash
docker compose build
```

Run:

```bash
docker compose up
```

Open:

```
http://localhost:8501
```

---

# Technologies

- Python
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- ChromaDB
- BM25
- CrossEncoder
- Pandas
- Docker

---

# Current Limitations

- Uses a single SEC filing.
- Ingestion is currently script-based rather than orchestrated with a workflow tool.
- The monitoring dashboard stores interactions locally in CSV format.

---

# Future Improvements

- Automated ingestion with Prefect or Kestra
- Multiple SEC filings
- Cloud deployment
- Persistent monitoring database
- Authentication
- REST API
- Better query rewriting using an LLM

---

