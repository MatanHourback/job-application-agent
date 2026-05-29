# Job Application Agent — RAG Portfolio Project

An AI-powered agent that analyzes your resume against a job description, flags skill gaps, and drafts a tailored cover letter using Retrieval-Augmented Generation (RAG).

## Architecture

  Resume PDF + Job Description
        ↓
  PyMuPDF (text extraction)
        ↓
  RecursiveCharacterTextSplitter (chunking)
        ↓
  OpenAI Embeddings → ChromaDB (vector store)
        ↓
  LangChain RAG Chain (Phase 2)
        ↓
  Skill Gap Analysis + Cover Letter

## Project Structure

job-application-agent/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + endpoints
│   ├── parser.py        # PDF + text parsing
│   ├── embeddings.py    # Chunking + ChromaDB indexing
│   └── agent.py         # LangChain chains 
├── evals/ # Evaluation scripts 
│   ├── __init__.py
│   ├── test_cases.py
│   ├── judge.py
│   └── run_evals.py
├── frontend/
│   └── index.html       # UI
├── requirements.txt
├── .env.example
└── README.md

## Run the server then open the UI

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000` in your browser.

## Usage

1. Upload your resume PDF
2. Paste a job description
3. Click **Analyze** to get skill gaps + cover letter

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Document ingestion (parse, chunk, embed, store) |
| 2 | ✅ Next | LangChain RAG chains (skill gap + cover letter) |
| 3 | 🔜 | FastAPI refinement + error handling |
| 4 | 🔜 | Evals with RAGAS (faithfulness, relevance, recall) |

## Eval Metrics

| Metric | Description |
|--------|-------------|
| Skill gap recall | % of known gaps correctly identified |
| Cover letter faithfulness | Are all claims grounded in the resume? |
| Cover letter relevance | Does the letter address the job description? |
| Latency | End-to-end response time |

## Design Decisions

- **Chunk size 500, overlap 50**: Resumes are dense but short. Smaller chunks = more precise retrieval.
- **ChromaDB**: Local, no account needed — perfect for development and portfolios.
- **`text-embedding-3-small`**: Cost-effective, high quality for English text.
- **RecursiveCharacterTextSplitter**: Prefers splitting on paragraphs before words, preserving context.
