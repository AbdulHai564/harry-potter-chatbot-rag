# 🪄 Harry Potter RAG Chatbot

A RAG chatbot that answers questions in the voice of your favorite Harry Potter characters — powered by parent-child chunking, Qdrant Cloud, and Groq.

**Live demo:** [harry-potter-chatbot-rag-btyvwkvrtrdkcvoywjthvc.streamlit.app](https://harry-potter-chatbot-rag-btyvwkvrtrdkcvoywjthvc.streamlit.app/)

## What it does

Ask a question about the Harry Potter universe, pick a character (Harry, Hermione, Dumbledore, Snape, Voldemort, and more), and get an answer written in that character's voice — grounded in the actual text of all 7 books.

## How it works

Instead of standard flat-chunk RAG, this project uses **Parent-Child chunking** via LangChain's `ParentDocumentRetriever`:

- **Child chunks** (400–500 chars) are embedded and stored in **Qdrant Cloud** — small enough for precise semantic search.
- **Parent chunks** (2000 chars) are stored in a persistent **LocalFileStore** on disk — when a child chunk matches a question, its larger parent chunk is returned instead, giving the LLM richer context without sacrificing search precision.
- No reranking or hybrid search needed — for narrative text like novels, semantic search on the child chunks is accurate enough on its own.

**Flow:**
```
PDF (7 books) → PyPDFLoader → parent/child splitters
              → child chunks → Qdrant Cloud (vector search)
              → parent chunks → LocalFileStore (disk persistence)
              → ParentDocumentRetriever ties them together
              → retrieved context + character persona → Groq LLM → answer
```

## Tech stack

- **LangChain** — `ParentDocumentRetriever`, document loading & splitting
- **Qdrant Cloud** — vector storage for child chunks
- **HuggingFace `all-MiniLM-L6-v2`** — embeddings
- **Groq (`openai/gpt-oss-120b`)** — LLM inference
- **Streamlit** — frontend
- **FastAPI** — REST API alternative (`api.py`)

## Characters available

Harry Potter, Hermione Granger, Dumbledore, Ron Weasley, Draco Malfoy, Severus Snape, Voldemort, Hagrid, Luna Lovegood, Sirius Black — each with their own personality prompt.

## Running locally

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with:
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key
COLLECTION_NAME=harry_potter
```

3. Run the app:
```bash
streamlit run app.py
```

First run will index the PDF into Qdrant (one-time cost) and cache the retriever for the session.

## API

A FastAPI version is also available:
```bash
uvicorn api:app --reload
```
- `GET /characters` — list available characters
- `POST /ask` — `{"question": "...", "character": "..."}` → returns an in-character answer

## Project structure

```
├── app.py          # Streamlit frontend
├── api.py          # FastAPI backend
├── retrieval.py     # Retriever setup, character personas, answer generation
├── ingestion.py     # PDF loading & chunking
├── config.py        # Env/secrets loading
├── reports/          # Source PDF
└── requirements.txt
```

## Related project

This project builds on lessons learned from [FinanceIQ](https://github.com/AbdulHai564/FinanceIQ), an earlier RAG chatbot using hybrid search + Cohere reranking.
