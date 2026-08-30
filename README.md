# AeroIntel

AeroIntel is an aviation-focused AI engineering and safety intelligence platform.

The project is being built from first principles to explore and implement:

- text embeddings
- semantic search
- vector retrieval
- keyword and hybrid retrieval
- retrieval-augmented generation (RAG)
- grounded answers with citations
- aviation safety data analysis

## Current Status

AeroIntel is currently in Stage 0: Local AI Foundation.

Implemented so far:

- Local LLM inference with Ollama
- Qwen3 local generation
- EmbeddingGemma embeddings
- 768-dimensional text embeddings
- Cosine similarity implemented in Python
- Basic semantic-search ranking experiment

## Current Semantic Search Flow

```text
Query text
    ↓
EmbeddingGemma
    ↓
Query embedding

Candidate text
    ↓
EmbeddingGemma
    ↓
Document embeddings

Query embedding
    ↓
Cosine similarity
    ↓
Ranked semantic search results
```

## Phase 0 — API Skeleton (current)

The script pipeline is now a real HTTP service. Same proven behavior
(distance-ceiling refusal, forced [n] citations, 768-dim validation),
behind a versioned API.

```text
POST /v1/query    question -> embed -> pgvector -> distance gate -> Qwen3
                  -> cited answer, or structural refusal (LLM never called)
POST /v1/search   semantic search only, no generation
POST /v1/ingest   embed + store one chunk
GET  /health/live    process is up
GET  /health/ready   database is reachable
```

### Architecture

```text
client -> FastAPI (uvicorn)
              ├── schemas/      pydantic request/response contracts
              ├── services/     retrieval, grounded generation, ingestion
              ├── llm/          Ollama client: timeout + retry + circuit breaker
              ├── db/           SQLAlchemy 2.0 async + pgvector
              └── core/         config (pydantic-settings), JSON logs, resilience
postgres (pgvector)   redis (Phase 2+)      ollama (on host)
```

### Run it from scratch

Prerequisites: Python 3.11, Docker Desktop, Ollama with `embeddinggemma`
and `qwen3:4b-instruct` pulled.

```powershell
# 0. One-time: create the virtualenv and install dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

# 1. Configure: copy .env.example to .env and set POSTGRES_PASSWORD
Copy-Item .env.example .env   # then edit .env

# 2. Start the stack (Postgres + Redis + API)
docker compose up -d --build

# 3. Apply schema migrations (first time only, from the host)
alembic upgrade head

# 4. Check it is alive (curl.exe, NOT curl - in PowerShell "curl" is an
#    alias for Invoke-WebRequest with different flags)
curl.exe http://localhost:8000/health/live     # {"status":"ok"}
curl.exe http://localhost:8000/health/ready    # {"status":"ready"} = DB reachable
```

Interactive API docs: http://localhost:8000/docs

### Try the API

Easiest: the interactive docs at http://localhost:8000/docs ("Try it out").
Or with curl.exe, putting JSON in a temp file to avoid PowerShell quoting
pain:

```powershell
'{"question":"engine shaking while ascending"}' | Set-Content q.json
curl.exe -X POST http://localhost:8000/v1/query -H "Content-Type: application/json" -d "@q.json"
del q.json
```

### Logs

```powershell
docker compose logs -f api        # live JSON logs with request IDs
docker compose ps                 # container status/health
docker compose logs postgres      # database logs
```

### Tests

```powershell
pytest -m "not local" -v   # 11 tests: resilience, API contract, pgvector ranking
pytest -m local            # 2 full RAG tests, need real Ollama (~30s)
ruff check app tests       # lint
ruff format app tests      # auto-format
mypy app                   # type check
```

CI (GitHub Actions) runs lint (ruff, mypy) and the non-local tests with a
pgvector service container on every push.

### Stop / reset

```powershell
docker compose down          # stop everything (data persists in the volume)
docker compose down -v       # stop AND wipe the database volume
```
