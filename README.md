# AeroIntel

Aviation maintenance RAG (retrieval-augmented generation), built locally, stage by
stage. Every stage is proven by its own verify script before the next one starts.
This isn't a demo wired together in an afternoon — it's a real system, grown one
verified layer at a time, from raw embeddings to a versioned HTTP API to (eventually)
a Kubernetes deployment with observability and hardening.

**Core rule, unchanged since Stage 2:** the model never answers from its own
knowledge. It only sees numbered context blocks pulled from the database, and if
nothing in the database is close enough to the question, the model is never even
called — the refusal is structural, enforced by code, not by asking nicely in a
prompt.

## Status

| Stage / Phase | What it proved | State |
|---|---|---|
| Stage 0 — local inference | Ollama running Qwen3 (generation) and EmbeddingGemma (768-dim embeddings) on the host | ✅ verified |
| Stage 1 — pgvector persistence | Embeddings round-trip through Postgres/pgvector correctly (8/8 checks) | ✅ verified |
| Stage 1D — semantic ranking | Cosine search ranks by actual meaning, not luck (6/6 checks) | ✅ verified |
| Stage 2 — cited generation + refusal | Distance-ceiling gate + forced `[n]` citations + structural refusal (6/6 checks) | ✅ verified |
| Phase 0 — application skeleton | Same RAG behavior, now a real FastAPI service: async SQLAlchemy + Alembic, timeout/retry/circuit-breaker around Ollama, structured JSON logs, Docker, CI | ✅ verified |
| Phase 1 — hybrid retrieval + eval | BM25 + pgvector merged via Reciprocal Rank Fusion, cross-encoder reranker, prompt registry, embedding-version tracking, golden-set eval harness (recall@k, MRR, faithfulness) in CI | ✅ built |
| Phase 2 — production API layer | Redis rate limiting, JWT auth, idempotency keys, SSE streaming, semantic response cache | not started |
| Phase 3 — event-driven ingestion | Redis Streams saga, dead-letter queue, outbox pattern, scheduled re-indexing | not started |
| Phase 4 — data layer at scale | Real PDF ingestion (page-level citations), HNSW index, pgBouncer, read replica, MinIO | not started |
| Phase 5 — deploy & infrastructure-as-code | Local `kind` Kubernetes cluster, Helm charts, Terraform, canary/blue-green deploys | not started |
| Phase 6 — observability | Prometheus, Grafana, OpenTelemetry traces, Jaeger, k6 load tests, SLO/error-budget dashboard | not started |
| Phase 7 — security & hardening | Keycloak OIDC, TLS via mkcert, SOPS-encrypted secrets, audit log, OWASP Top-10 pass | not started |

> **Full Phase 0 walkthrough:** [docs/PHASE_0.md](docs/PHASE_0.md) — what every
> file does and why, concept by concept.

### A note on the data

The database currently holds **6 hand-written synthetic chunks** (hydraulics,
pressurization, brakes, engine) — see `scripts/load_sample_data.py`. This is
placeholder content written for this project, explicitly **not** real aircraft
manual text and **not** airworthiness information. Real PDF ingestion is Phase 4
work. Everything through Phase 1 is built and measured against this small
synthetic set.

## How a request flows (current: Phase 0 + Stage 2)

```text
POST /v1/query {"question": "..."}
        │
        ▼
  embed question (EmbeddingGemma, via Ollama)
        │
        ▼
  pgvector cosine search over document_chunks, top_k closest
        │
        ▼
  distance ≤ MAX_DISTANCE (0.48) ?  ── no ──▶  refuse, LLM never called
        │ yes
        ▼
  build numbered CONTEXT blocks [1] [2] ...
        │
        ▼
  Qwen3 generates, forced to cite blocks like [1] or [2, 3]
        │
        ▼
  QueryResponse { answer, chunks, refused }
```

Every Ollama call (`embed` and `chat`) is wrapped in timeout → retry-with-backoff
→ circuit-breaker, so a slow or dead Ollama fails fast and predictably instead of
hanging the whole API.

## Architecture

```text
client
  │  HTTP
  ▼
RequestIdMiddleware        stamps + logs a request ID on every request
  │
  ▼
FastAPI router              /v1/query  /v1/search  /v1/ingest
                             /health/live  /health/ready
  │
  ▼
services/                   retrieval.py · generation.py · ingestion.py
  │                         (the RAG logic — no HTTP concerns)
  ├──────────────┬──────────────────────
  ▼              ▼
db/              llm/
SQLAlchemy       OllamaClient
2.0 async,       timeout + retry + circuit breaker,
Alembic          wraps embed() and chat()
  │              │
  ▼              ▼
Postgres         Ollama
(pgvector)       (Qwen3, EmbeddingGemma — runs on the host)
```

`core/` holds the cross-cutting pieces: `config.py` (typed settings from env/`.env`,
no defaults for secrets), `logging.py` (structlog JSON + request-ID middleware),
`resilience.py` (the retry + circuit-breaker primitives).

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI, Uvicorn, pydantic-settings |
| Data | PostgreSQL 16, pgvector, SQLAlchemy 2.0 (async), Alembic |
| AI / retrieval | Ollama, Qwen3 (generation), EmbeddingGemma (768-dim embeddings) |
| Resilience | Hand-rolled retry + circuit breaker (`app/core/resilience.py`), httpx timeouts |
| Ops | structlog (JSON logs + request IDs), Docker, Docker Compose, GitHub Actions CI |
| Test / lint | pytest, pytest-asyncio, ruff, mypy |
| Messaging (provisioned, unused until Phase 2+) | Redis |

## Project structure

```text
app/
  main.py              app factory + lifespan (startup/shutdown)
  core/
    config.py           typed Settings, env/.env driven
    logging.py           structlog JSON logs, RequestIdMiddleware
    resilience.py         retry() with backoff, CircuitBreaker
  api/
    deps.py              FastAPI dependencies (DB session, Ollama client)
    health.py             /health/live, /health/ready
    v1/router.py           /v1/query, /v1/search, /v1/ingest
  db/
    base.py               async engine + session factory
    models.py              DocumentChunk (pgvector column, JSONB metadata)
  llm/
    ollama.py              OllamaClient: embed() + chat(), resilience-wrapped
  schemas/                 pydantic request/response contracts
  services/
    retrieval.py            search_chunks() — embed + pgvector search
    generation.py            answer_question() — distance gate + grounded generation
    ingestion.py             ingest_chunk() — embed + store

alembic/                  versioned schema migrations
scripts/                  original stage scripts (ask.py, verify_*.py, ...) — still work
tests/                    pytest suite (unit, integration, local-Ollama)
sql/                      original raw-SQL schema (pre-Alembic reference)
docs/PHASE_0.md           file-by-file walkthrough of Phase 0
.github/workflows/ci.yml  lint + test on every push
docker-compose.yml        postgres (pgvector) + redis + api
Dockerfile                multi-stage, non-root user
```

## Run it from scratch

Prerequisites: Python 3.11, Docker Desktop, Ollama with `embeddinggemma` and
`qwen3:4b-instruct` pulled.

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

# 4. (Optional) load the 6 synthetic sample chunks
python scripts/load_sample_data.py

# 5. Check it is alive (curl.exe, NOT curl — in PowerShell "curl" is an
#    alias for Invoke-WebRequest with different flags)
curl.exe http://localhost:8000/health/live     # {"status":"ok"}
curl.exe http://localhost:8000/health/ready    # {"status":"ready"} = DB reachable
```

Interactive API docs: http://localhost:8000/docs

## API reference

| Endpoint | What it does |
|---|---|
| `POST /v1/query` | Full grounded pipeline: embed → search → distance gate → generate with citations, or a structural refusal. Refusal is a normal `200` (`"refused": true`) — a successful outcome of the contract, not an error. |
| `POST /v1/search` | Semantic search only. No generation, no LLM call. |
| `POST /v1/ingest` | Embed and store one chunk of text with metadata. |
| `GET /health/live` | Is the process up? Checks nothing — always `200` if alive. |
| `GET /health/ready` | Can we serve traffic? Verifies the database is reachable; `503` if not. |

### Try it

Easiest: the interactive docs at http://localhost:8000/docs ("Try it out"). Or with
`curl.exe`, putting JSON in a temp file to avoid PowerShell quoting pain:

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

## Phase 1: retrieval quality

Both retrieval upgrades are feature-flagged and **off by default**, so the
system behaves exactly as it did in Phase 0 until you turn them on.

| Setting | Default | What it does |
|---|---|---|
| `HYBRID_ENABLED` | `false` | Runs BM25 keyword search alongside vector search and merges the two rankings with Reciprocal Rank Fusion. Catches exact terms (part numbers, `2.5 bar`) that embeddings generalise away. |
| `RERANKER` | `none` | `crossencoder` re-scores the top `RERANK_CANDIDATES` with a model that reads query and chunk together. Needs `pip install -r requirements-rerank.txt`. |
| `PROMPT_VERSION` | unset | Pins a prompt version (e.g. `v1`); unset uses the newest file in `app/prompts/`. |

### Prompts are versioned files

Grounding instructions live in `app/prompts/<name>.<version>.txt`, not as
string literals. Rewording means adding `v2.txt` — a reviewable diff, with
`v1` still runnable beside it. Each prompt carries a sha256 of its bytes,
so editing `v1` in place is detectable: the checksum moves, and last
month's eval numbers are visibly no longer comparable rather than quietly
misleading. Every answer, log line and eval report names the exact prompt
that produced it (`grounded_answer@v1:2118f53d`).

### Embedding version tracking

Every row records the model that produced its vector in `embedding_model`.
Vectors from different models are not comparable, so a model swap is a
deliberate migration, not an accident:

```powershell
python scripts/backfill_embeddings.py --dry-run          # what would change
python scripts/backfill_embeddings.py                    # tag untagged rows
python scripts/backfill_embeddings.py --model newmodel   # re-embed under a new model
```

Rows keep their old tag until re-embedded, so a crash midway leaves a
visible mix rather than a silently corrupted index. Flip `EMBEDDING_MODEL`
only after a clean run.

### Measuring retrieval quality

```powershell
python -m evals.run_eval                                   # vector baseline
python -m evals.run_eval --retriever hybrid                # + BM25/RRF
python -m evals.run_eval --retriever hybrid --reranker crossencoder
python -m evals.run_eval --faithfulness                    # + hallucination check
```

Reports `recall@k` and `MRR` against `evals/golden_set.json`, and exits
non-zero if `recall@5` falls below the threshold. `--faithfulness` also
generates an answer per question and grades it two ways:

- **Citation validity** — deterministic and free: every `[n]` the answer
  cites must point at a CONTEXT block that was actually supplied. A model
  citing `[4]` when handed three blocks invented a source, and the run
  fails. This is the check safe to gate on.
- **Judge faithfulness** — an LLM reads (context, answer) and rules on
  whether every claim is supported. It is itself a fallible model, so
  treat a drop as a prompt to go read the answers, not a number to
  optimise blindly. An unparseable verdict scores `None`, not `0` — a
  broken judge must not be recorded as a hallucinating model.

CI cannot measure real quality (no Ollama on the runner), so it guards the
**harness** instead: `tests/test_eval_harness.py` drives the whole scoring
path against Postgres with deterministic stub vectors. A silently broken
scoreboard is how a regression reaches main unnoticed.

## Tests

```powershell
pytest -m "not local" -v   # unit + API contract + pgvector ranking (runs in CI)
pytest -m local            # full RAG tests, need real Ollama (~30s)
ruff check app tests       # lint
ruff format app tests      # auto-format
mypy app                   # type check
```

CI (`.github/workflows/ci.yml`) runs lint (ruff + mypy) and the non-local tests
against a real pgvector service container on every push.

| Test file | What it proves | Needs |
|---|---|---|
| `tests/test_resilience.py` | Retry recovers from transient failures, gives up after the last attempt; breaker opens after N failures, fails fast while open, recovers via a successful probe | nothing — pure asyncio |
| `tests/test_health.py` | Liveness works, every response carries `x-request-id`, empty questions are rejected with `422` before touching services | nothing |
| `tests/test_search_synthetic.py` | pgvector ranks by meaning using orthogonal synthetic vectors — no real embedding model needed to prove the math | Postgres only (CI) |
| `tests/test_rag_local.py` | Full grounding contract with real Ollama: matched question → cited answer, unmatched question → refusal | Ollama + Postgres |
| `tests/test_eval_metrics.py` | Eval metric math (recall@k, MRR, aggregation) + golden-set well-formedness | nothing |

## Retrieval evals (the scoreboard)

Before and after any retrieval change, run the eval harness and compare
numbers — never judge retrieval quality by eyeballing a few queries:

```powershell
python -m evals.run_eval   # needs Postgres (with sample data) + Ollama
```

It runs every question in `evals/golden_set.json` through the live
retrieval pipeline and reports recall@1/3/5 and MRR per question, plus
means. The eval exits 1 if mean recall@5 drops below 0.9.

**Baseline (vector-only retrieval, 12-question golden set):**

```text
recall@1=0.667  recall@3=1.000  recall@5=1.000  MRR=0.833
```

**Hybrid (BM25 + pgvector + RRF, `--retriever hybrid`):**

```text
recall@1=0.667  recall@3=1.000  recall@5=1.000  MRR=0.833
```

An honest, instructive result: on this corpus hybrid changes nothing. The
four rank-2 misses are paraphrase questions where the competing chunk is
*semantically* close (same system, different section) — a ranking problem,
not a recall problem. Keyword overlap can't fix that; a cross-encoder
reranker reading the full text can. That is step 3. Hybrid stays in the
codebase behind `HYBRID_ENABLED` because it will matter once the corpus
grows to contain exact-term content (error codes, part numbers) that
embeddings miss.

## Stop / reset

```powershell
docker compose down          # stop everything (data persists in the volume)
docker compose down -v       # stop AND wipe the database volume
```

## Roadmap detail

The full phase-by-phase build plan — including every concept each phase teaches
(hybrid retrieval, circuit breakers, HNSW indexing, blue-green deploys, SLOs, OIDC,
and more) — lives in the AeroIntel project's **Build Plan** document. The table
under [Status](#status) above is the short version.
