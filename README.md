# Financial RAG Assistant

A Retrieval-Augmented Generation (RAG) application for answering questions about SEC financial filings using hybrid retrieval, document reranking, local LLM generation, and automated ingestion workflows.

The application retrieves relevant sections from company filings and generates grounded answers using only retrieved documents.

---

# Project Objective

Financial reports are lengthy and difficult to search manually.

This project builds a RAG system that allows users to ask natural language questions about SEC filings while ensuring responses are grounded in the original filing.

Example questions:

* What were Apple's total net sales in 2024?
* What cybersecurity risks does Apple disclose?
* What are Apple's operating segments?
* What supply chain risks are discussed?

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

* Hybrid retrieval (BM25 + Vector Search)
* Query rewriting
* Cross-Encoder document reranking
* Local LLM generation using Hugging Face Transformers
* Streamlit interface
* Monitoring dashboard
* User feedback collection
* Response time tracking
* Docker support
* Prefect workflow orchestration
* Automated ingestion pipeline
* Retrieval evaluation
* LLM evaluation

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
├── workflows/
│   └── ingestion_flow.py
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
│   └── logger.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
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

# Data Ingestion Workflow

The ingestion pipeline is orchestrated using Prefect.

The workflow automates:

1. PDF text extraction
2. Document chunking
3. Embedding generation
4. ChromaDB vector database creation

Run the ingestion workflow:

```bash
python workflows/ingestion_flow.py
```

Generated outputs:

```
data/processed/apple_report.txt
data/processed/apple_chunks.json
data/embeddings/chroma_db/
```

Prefect provides workflow execution tracking, task logging, and pipeline visibility.

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

This improves retrieval recall.

---

# Hybrid Retrieval

The project combines:

* BM25 keyword search
* SentenceTransformer embeddings
* ChromaDB vector similarity

Hybrid retrieval improves both precision and recall compared with individual retrieval methods.

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

* only retrieved context is used
* no hallucinations
* no external knowledge
* financial answers remain grounded in the filing

---

# Evaluation

## Retrieval Evaluation

Implemented in:

```
evaluation/evaluate_retrieval.py
```

Evaluated approaches:

* Vector Search
* BM25
* Hybrid Search
* Hybrid + Reranker

---

## LLM Evaluation

Implemented in:

```
evaluation/evaluate_llm.py
```

Compared prompting strategies:

* Baseline Prompt
* Production Prompt

Example results:

| Prompt     | Average Score |
| ---------- | ------------- |
| Baseline   | 0.125         |
| Production | 1.00          |

---

# Monitoring Dashboard

A Streamlit dashboard tracks system performance.

Collected metrics:

* Total questions
* Positive feedback
* Negative feedback
* Average response time

Charts:

* Feedback distribution
* Questions over time
* Response time trend
* Response time distribution
* Question length distribution

---

# Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app/main.py
```

Open:

```
http://localhost:8501
```

---

# Running the Ingestion Workflow

Install Prefect:

```bash
pip install prefect
```

Run:

```bash
python workflows/ingestion_flow.py
```

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

* Python
* Streamlit
* Hugging Face Transformers
* Sentence Transformers
* ChromaDB
* BM25
* CrossEncoder
* Prefect
* Pandas
* Docker

---

# Current Limitations

* Supports a single SEC filing.
* Prefect workflow currently runs locally.
* Monitoring data is stored locally in CSV format.

---

# Future Improvements

* Scheduled Prefect deployments for automatic SEC filing ingestion
* Multiple SEC filings support
* Cloud deployment
* Persistent monitoring database
* Authentication
* REST API
* LLM-based query rewriting

---
