# Job Application Agent — RAG Portfolio Project

An AI-powered agent that analyzes your resume against a job description, flags skill gaps, and drafts a tailored cover letter using Retrieval-Augmented Generation (RAG).

## Architecture

```
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
```

## Project Structure

```
job-application-agent/
├── app/
│   ├── main.py          # FastAPI app + endpoints
│   ├── parser.py        # PDF + text parsing
│   ├── embeddings.py    # Chunking + ChromaDB indexing
│   └── agent.py         # LangChain chains 
├── frontend/
│   └── index.html       # UI
├── evals/               # Evaluation scripts 
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd job-application-agent
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

### 4. Open the UI

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

## Eval Metrics (Phase 4)

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
