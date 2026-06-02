# Job Application Agent

An AI-powered agent that analyzes your resume against a job description, flags skill gaps, and drafts a tailored cover letter using Retrieval-Augmented Generation (RAG).

## What It Does

1. Upload a **resume PDF** and paste a **job description**
2. The agent parses, chunks, and indexes both into a local vector database
3. A LangChain RAG chain retrieves relevant context and identifies **skill gaps**
4. A second RAG chain generates a **tailored cover letter** grounded in the resume
5. An eval framework measures output quality across 5 cross-sector test cases

## Architecture

```
Resume PDF + Job Description
        ↓
  PyMuPDF (text extraction)
        ↓
  RecursiveCharacterTextSplitter (chunking)
  chunk size: 500, overlap: 50
        ↓
  OpenAI Embeddings (text-embedding-3-small)
        ↓
  ChromaDB (local vector store)
        ↓
  LangChain RAG Chain
  ├── Skill Gap Chain   → retrieves resume + JD chunks → GPT-3.5-turbo
  └── Cover Letter Chain → retrieves resume + JD chunks → GPT-3.5-turbo
        ↓
  FastAPI Backend → HTML/JS Frontend
        ↓
  Eval Framework
  ├── Recall         → deterministic, string matching
  ├── Faithfulness   → Claude Sonnet as judge
  └── Relevance      → Claude Sonnet as judge
```

## Project Structure

```
job-application-agent/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + endpoints
│   ├── parser.py        # PDF parsing + text cleaning
│   ├── embeddings.py    # Chunking, embedding, ChromaDB indexing
│   └── agent.py         # LangChain RAG chains
├── evals/
│   ├── __init__.py
│   ├── test_cases.py    # 5 labeled resume + JD pairs across sectors
│   ├── judge.py         # Claude-as-judge scoring functions
│   └── run_evals.py     # Eval loop + results logging
├── frontend/
│   └── index.html       # Dark-themed UI with drag-and-drop upload
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/MatanHourback/job-application-agent
cd job-application-agent
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Add your API keys to `.env`:
```
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

### 4. Open the UI

Visit `http://localhost:8000` in your browser.

---

## Usage

1. Upload your **resume PDF** using the drag-and-drop zone
2. Paste a **job description** into the text area
3. Click **Analyze Application**
4. Get back a skill gap analysis and a tailored cover letter in seconds

---

## Eval Framework

The eval framework tests output quality across 5 real-world test cases spanning completely different job sectors:

| Sector | Resume Type | Key Gaps Tested |
|---|---|---|
| Programming | Junior developer → Senior ML Engineer | Technical skills, seniority |
| Architecture | Architect → Design firm role | Software tools, specialization |
| Sales | SDR → SDR at new company | Behavioral, methodology gaps |
| Weapons Manufacturing | Aerospace engineer → Defense role | Security clearance, credentials |
| Healthcare | Medical assistant → Registered Nurse | Licensing, certification gaps |

### Metrics

**Recall** — what % of known skill gaps did the agent correctly identify?
Measured deterministically by checking if each known gap appears in the agent output.

**Faithfulness** — are cover letter claims grounded in the resume?
Measured using Claude Sonnet as judge (0-1 scale). Detects hallucinations.

**Relevance** — does the cover letter address the job description?
Measured using Claude Sonnet as judge (0-1 scale). Detects generic output.

### Results — Before and After Prompt Iteration

| Version | Change Made | Recall | Faithfulness | Relevance |
|---|---|---|---|---|
| v1 baseline | Initial run | 0.13 | 0.08 | 0.17 |
| v2 | Fixed ChromaDB stale data bug + improved prompts | 0.68 | 0.42 | 0.56 |
| v3 | Added grounding instruction to cover letter prompt | 0.68 | 0.45 | 0.47 |

### Key Finding

v1 scores were near zero due to a ChromaDB stale data bug — each test case was inheriting the vector store from the previous one. Identified through eval diagnostics (printing agent output per test case), fixed by clearing collections between runs.

After the fix, recall improved from 0.13 → 0.68. Faithfulness remains the weakest metric — the model hallucinates credentials for highly specialized roles (defense, healthcare) where the candidate is significantly underqualified. A production fix would add a post-generation verification step or a stronger retrieval grounding strategy.

### Run the Evals

```bash
python3 -m evals.run_evals
```
Results are saved to `evals/results.json`.

## Design Decisions

**Why manual retrieval over LangChain retrievers?**
The skill gap and cover letter chains both need context from two separate collections (resume + job description). LangChain's built-in retriever chains are designed for single-source retrieval, so manually fetching from both collections and merging into one prompt gave more control.

**Why ChromaDB?**
Local, no account needed, persists to disk. Ideal for development and portfolio demos without infrastructure overhead.

**Why chunk size 500, overlap 50?**
Resumes are dense but short. Smaller chunks produce more precise retrieval. Overlap of 50 ensures context isn't lost at chunk boundaries.

**Why GPT-3.5-turbo for generation and Claude Sonnet for judging?**
GPT-3.5-turbo is cost-effective for high-volume generation during development. Claude Sonnet was chosen as the judge because using a different model family to evaluate reduces bias — the judge isn't scoring its own outputs.

**Why deterministic recall instead of LLM-as-judge?**
Recall is objective — either the gap is mentioned or it isn't. Using an LLM to score something deterministic adds unnecessary variance and cost.

## What I'd Improve Next

- **Reranking** — add a cross-encoder reranker between retrieval and generation to improve chunk quality
- **Streaming** — stream the cover letter token by token for a better UX
- **Model comparison** — run evals with Claude Sonnet as the generator and compare faithfulness scores against GPT-3.5-turbo
- **Larger eval dataset** — 5 test cases is enough to demonstrate the concept; 20+ would give statistically meaningful scores
- **Deploy** — host on Railway or Render with a public URL

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11 |
| LLM | GPT-3.5-turbo (generation), Claude Sonnet (eval judge) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Parsing | PyMuPDF |
| Frontend | HTML, CSS, JavaScript |
| Eval Framework | Custom (recall + LLM-as-judge) |