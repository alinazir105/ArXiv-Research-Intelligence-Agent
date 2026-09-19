
# ArXiv Research Intelligence Agent

This repository contains a working research assistant that answers questions over ArXiv papers and web information using a local retrieval pipeline, a LangGraph agent, and an OpenAI-backed answer generator. It is designed for exploring AI/ML literature, summarizing topics, and retrieving specific papers from the ArXiv corpus while grounding answers in retrieved evidence.

The implementation in this repo is the current state of the project: a FastAPI service, a hybrid retriever over Qdrant, Redis caching, rate limiting, streaming responses, evaluation scripts, and a fine-tuning pipeline for a Phi-3-based academic generation setup.

---

## What the project does

- Answers research questions using a local ArXiv corpus
- Uses a LangGraph ReAct-style agent with tool calls for:
  - corpus search
  - web search
  - fetching a specific paper by URL
  - summarizing papers on a topic
- Uses hybrid retrieval combining:
  - dense vector search over Qdrant
  - BM25 sparse retrieval
  - reciprocal rank fusion (RRF)
  - cross-encoder reranking
- Uses HyDE and query decomposition to improve retrieval quality on academic prompts
- Streams generated answers token-by-token over SSE
- Caches repeated queries in Redis
- Includes an evaluation harness for faithfulness, relevancy, and tool selection accuracy
- Includes a fine-tuning workflow for creating a small academic Q&A dataset and training a LoRA adapter

---

## Current architecture

```text
User query
  |
  v
FastAPI app (/chat, /chat/stream, /health)
  |
  +-- Redis cache check
  |
  v
LangGraph agent (tool-calling ReAct loop)
  |
  +-- search_corpus -> HybridRetriever
  |      +-- query decomposition
  |      +-- HyDE (hypothetical abstract generation)
  |      +-- OpenAI embeddings -> Qdrant vector search
  |      +-- BM25 retrieval
  |      +-- RRF fusion
  |      +-- cross-encoder reranking
  |
  +-- search_web -> DuckDuckGo
  |
  +-- fetch_paper -> arXiv API
  |
  +-- summarize_papers -> corpus overview
  |
  v
OpenAI answer generation
  |
  v
Redis cache + API response
```

---

## Current implementation details

### API layer

The app is defined in [app/main.py](app/main.py). It exposes:

- `GET /` — simple health/root endpoint
- `POST /chat` — cached or uncached agent call
- `POST /chat/stream` — streamed answer response using SSE
- `GET /health` — checks Qdrant and retriever readiness

The app initializes the retriever on startup and applies a rate limit of 10 requests per minute per IP via `slowapi`.

### Agent layer

The tool-calling agent is implemented in [app/agent/agent.py](app/agent/agent.py). It uses:

- `search_corpus(query)` to retrieve relevant ArXiv chunks
- `search_web(query)` to search the web when current or external information is needed
- `fetch_paper(url)` to fetch a specific paper through the arXiv API
- `summarize_papers(query)` to retrieve and summarize a topic-level set of papers

The agent is built with LangGraph state graphs and `langchain_openai.ChatOpenAI`.

### Retrieval layer

The hybrid retrieval system is in [app/retrieval/retriever.py](app/retrieval/retriever.py). It does the following for each user query:

1. decomposes the query into focused sub-questions
2. generates a hypothetical academic abstract with HyDE
3. embeds the generated abstract using the configured OpenAI embedding model
4. runs vector search against Qdrant
5. runs BM25 search over the same corpus
6. merges both with reciprocal rank fusion
7. reranks the top candidates using a cross-encoder
8. returns the final top-k chunk results

This is the current retrieval strategy implemented in the codebase.

### Indexing and ingestion

The ingestion pipeline is split between abstract indexing and PDF-based contextual indexing:

- [scripts/pipeline.py](scripts/pipeline.py) indexes abstract-level chunks for seed research topics
- [scripts/run_pdf_pipeline.py](scripts/run_pdf_pipeline.py) downloads PDFs, extracts text with PyMuPDF, adds contextual metadata, and stores chunks in Qdrant
- [app/ingestion/ingest.py](app/ingestion/ingest.py) fetches papers from arXiv
- [app/ingestion/indexer.py](app/ingestion/indexer.py) embeds chunks and stores them in Qdrant
- [app/ingestion/pdf_processor.py](app/ingestion/pdf_processor.py) handles PDF extraction and contextual chunk generation

### Configuration and env

The project settings live in [app/core/config.py](app/core/config.py). The main runtime variables are:

```env
OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
COLLECTION_NAME=arxiv_papers
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=arxiv-research-agent
```

---

## Tech stack in this repo

| Layer | Current implementation |
|---|---|
| API | FastAPI |
| Agent orchestration | LangGraph |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Reranker | `sentence-transformers` cross-encoder |
| Cache | Redis |
| Web search | `duckduckgo-search` |
| ArXiv access | `arxiv` Python client |
| PDF extraction | PyMuPDF |
| Rate limiting | `slowapi` |
| Pydantic config | `pydantic-settings` |
| Fine-tuning | PEFT + TRL + BitsAndBytes |

---

## Setup

### Prerequisites

- Python 3.12+
- Docker Desktop
- OpenAI API key
- Optional: LangSmith API key for tracing

### 1. Clone the repo

```bash
git clone https://github.com/alinazir105/arxiv-research-agent
cd arxiv-research-agent
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=arxiv-research-agent
```

If you want to override the default Qdrant and Redis endpoints, set:

```env
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

### 5. Start supporting services

Start Docker containers for Qdrant and Redis:

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
docker run -d --name redis -p 6379:6379 redis:alpine
```

If they already exist:

```bash
docker start qdrant
docker start redis
```

### 6. Build the corpus

Index seed-topic abstract data:

```bash
python -m scripts.pipeline
```

Index full-paper chunks with contextual metadata:

```bash
python -m scripts.run_pdf_pipeline
```

This is a longer-running step and pulls additional paper PDFs from ArXiv.

### 7. Start the app

```bash
uvicorn app.main:app --reload
```

The service will be available at:

- http://localhost:8000
- Swagger docs: http://localhost:8000/docs

---

## API endpoints

### `GET /health`

Returns the status of the Qdrant connection and the retriever initialization.

### `POST /chat`

Request body:

```json
{
  "query": "What are the main approaches to RAG evaluation?"
}
```

Response:

```json
{
  "answer": "...",
  "cached": false
}
```

### `POST /chat/stream`

Streams the answer as server-sent events. This is used for token-by-token output in the frontend or any streaming client.

---

## Evaluation workflow

The project includes an evaluation script in [tests/evaluate.py](tests/evaluate.py). It tests a set of corpus and web questions and scores:

- faithfulness
- answer relevancy
- tool selection accuracy

Run it with:

```bash
python -m tests.evaluate
```

This is a lightweight LLM-as-a-judge harness, not a formal benchmark suite, but it is useful for checking the agent's behavior over realistic queries.

---

## Fine-tuning workflow

The repo also contains a fine-tuning pipeline for academic-style generation.

### Generate the dataset

```bash
python -m finetune.generate_dataset
```

This reads chunks from Qdrant and creates a JSONL question/answer dataset for fine-tuning.

### Train the LoRA adapter

```bash
finetune_env\Scripts\activate
python finetune/train.py
```

The training code in [finetune/train.py](finetune/train.py) uses:

- `datasets`
- `transformers`
- `bitsandbytes`
- `peft`
- `trl`

It loads `microsoft/Phi-3-mini-4k-instruct`, applies 4-bit quantization, and trains a LoRA adapter for the generated dataset.

### Inference

```bash
python finetune/inference.py
```

---

## Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 0.750 |
| Answer Relevancy | 0.900 |
| Tool Selection | 10/10 (100%) |

---

## Project structure

```text
arxiv-research-agent/
├── app/
│   ├── agent/
│   │   └── agent.py          # LangGraph agent + tool definitions
│   ├── core/
│   │   ├── cache.py          # Redis cache helpers
│   │   ├── config.py         # Pydantic settings + .env loading
│   │   └── logger.py         # structured logging
│   ├── ingestion/
│   │   ├── chunker.py        # text chunking utilities
│   │   ├── indexer.py        # embedding + Qdrant upsert logic
│   │   ├── ingest.py         # ArXiv paper fetching
│   │   └── pdf_processor.py  # PDF extraction and contextual chunking
│   ├── models/
│   │   └── chat_request.py   # request validation
│   ├── retrieval/
│   │   └── retriever.py      # hybrid dense + sparse + rerank retrieval
│   └── main.py               # FastAPI app
├── finetune/
│   ├── generate_dataset.py   # convert Qdrant chunks into QA pairs
│   ├── inference.py          # LoRA adapter inference
│   ├── train.py              # training script for Phi-3-mini
│   └── output/               # saved adapter output
├── scripts/
│   ├── pipeline.py           # abstract indexing pipeline
│   └── run_pdf_pipeline.py   # full PDF indexing pipeline
├── tests/
│   └── evaluate.py           # evaluation harness
├── finetune_env/             # separate training environment
├── .env                      # secrets (not committed)
├── requirements.txt
├── README.md
└── logs/
```

---

## Notes on the current repo

This repo is a real working prototype rather than a static demo. The current implementation includes the end-to-end retrieval pipeline, the agent, the API, the evaluation harness, and the training tooling needed to explore the full stack of a research-agent system.

The project is best understood as a practical engineering exercise in:

- retrieval design
- agent/tool orchestration
- grounded generation
- dense + sparse retrieval fusion
- Qdrant vector search
- streaming API design
- LLM evaluation
- LoRA fine-tuning on a research corpus

It is intentionally focused on the ArXiv + research-knowledge workflow, not a generalized production platform for every domain.
 
