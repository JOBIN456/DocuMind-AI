# DocuMind AI 🧠

> Chat with any PDF using an agentic RAG pipeline with relevance grading, query rewriting, and hallucination detection.

---

## What is DocuMind AI?

DocuMind AI is an intelligent document question-answering system. Upload any PDF — structured reports, unstructured documents, manuals, or books — and chat with it using natural language.

It goes beyond basic RAG. The pipeline grades retrieved chunks for relevance, rewrites vague queries to improve retrieval, and validates answers against the source document to catch hallucinations before they reach you.

---

## Features

- 📄 Handles any PDF — structured or unstructured
- 🔍 Semantic search with Sentence Transformers
- 🧠 Relevance grading — filters out irrelevant chunks
- 🔄 Query rewriting — improves retrieval on vague questions
- ✅ Hallucination detection — answers are grounded in the document
- ⚡ Fast LLM inference via Groq
- 🐳 Qdrant vector database running on Docker
- 🗂️ Per-PDF collections — each document is isolated

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Pipeline | LangGraph |
| PDF Parsing | Docling |
| Chunking | HybridChunker |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | Qdrant (Docker) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Frontend | HTML / CSS / JS |

---

## Pipeline Architecture

```
PDF Upload
    ↓
Docling (parse + chunk)
    ↓
Sentence Transformers (embed)
    ↓
Qdrant (store vectors)

Chat
    ↓
Embed question
    ↓
Retrieve top chunks
    ↓
Grade relevance ──── irrelevant ──→ Rewrite query (max 2x)
    ↓ relevant
Build prompt
    ↓
Groq LLM
    ↓
Hallucination check ── hallucinated ──→ Retry
    ↓ grounded
Return answer
```

---

## Project Structure

```
documind-ai/
├── api/
│   ├── main.py          # FastAPI app entry point
│   ├── router.py        # PDF ingest + chat routes
│   ├── ingest.py        # LangGraph ingest pipeline
│   └── rag.py           # LangGraph RAG pipeline
├── db/
│   └── qdrant_setup.py  # Shared Qdrant client
├── templates/
│   ├── index.html       # Upload page
│   └── chat.html        # Chat page
├── docker-compose.yml   # Qdrant Docker setup
├── rag-flow
├── .gitignore
└── README.md
```


Built with ❤️ using FastAPI, LangGraph, Docling, and Qdrant.