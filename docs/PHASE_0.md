# Phase 0 — Application Skeleton: What We Built and Why

Phase 0 turned AeroIntel from a set of scripts (`ask.py`, `search_embeddings.py`)
into a real HTTP service. The RAG behavior did not change — same distance
ceiling, same citations, same refusal. What changed is *how the code is
organized*: each concern now lives in its own file with one job.

This document explains every file, one by one.

---

## The big picture

Before Phase 0:

```text
ask.py  = one script: connect to DB, embed, search, prompt, generate, print
```

After Phase 0:

```text
HTTP request
    → FastAPI (app/main.py)          receives, validates, routes
        → schemas/                    validates the request shape
        → services/                   the actual RAG logic
            → llm/ollama.py           talks to Ollama (with resilience)
            → db/                     talks to Postgres via SQLAlchemy
        → core/                       config, logging, resilience helpers
```

Why split it this way? Each layer can be tested alone, replaced alone, and
understood alone. That is the difference between a demo and a system.

---

## Configuration

### `app/core/config.py`
One `Settings` class holds every config value: database credentials, Ollama
URL, model names, `MAX_DISTANCE`, timeouts. It reads environment variables
and your `.env` file automatically and validates types.

- **Why one class?** The old scripts scattered `os.environ[...]` lookups
  everywhere. Now there is exactly one place config comes from.
- **Secrets have no defaults.** `postgres_password: str` has no default, so
  the app refuses to start without it — a missing secret should be a loud
  crash, not a silent placeholder.
- `get_settings()` is cached with `@lru_cache` so the whole app shares one
  instance.
- Two URL properties: `database_url` (async, for the app) and
  `sync_database_url` (for Alembic migrations, which run synchronously).

**Concept: configuration management.** Config is typed, validated at
startup, and read from one place — never scattered `os.environ` calls.

---

## Logging

### `app/core/logging.py`
Two things:

1. `configure_logging()` — sets up **structlog** so every log line is JSON
   with named fields (`request_id`, `level`, `timestamp`, `event`) instead
   of prose. Machines can filter JSON; humans grep prose.
2. `RequestIdMiddleware` — every HTTP request gets a unique `request_id`
   (or honors an incoming `X-Request-ID` header). It is stored in a
   *context variable*, so every log line emitted while handling that
   request automatically carries it, and it is echoed back in the response
   header.

**Concepts: structured logging, correlation IDs.** When one request
produces fifty log lines, you can pull them all out with one filter.

---

## Resilience

### `app/core/resilience.py`
Two small tools that keep the API alive when Ollama misbehaves:

- `retry(fn)` — calls an async function; on failure waits 0.5s, then 1s,
  then 2s (**exponential backoff**) before giving up. Transient hiccups
  heal themselves; you never hammer a struggling service.
- `CircuitBreaker` — tracks consecutive failures of a downstream service.
  After N failures it goes **open** and every call fails *instantly* with
  `CircuitOpenError` instead of piling onto something already dead. After
  a cool-down it goes **half-open**: one probe call decides whether to
  close (recovered) or re-open.

**Concepts: timeouts, retries with exponential backoff, circuit breakers.**
Timeouts live in the Ollama client (httpx's `Timeout`); these two live here.

---

## Database

### `app/db/base.py`
Creates one async SQLAlchemy **engine** (which holds a real connection
pool) and a session factory. `get_session()` is a FastAPI dependency that
gives each request its own session and closes it afterwards.

**Concept: connection pooling.** The scripts opened a fresh database
connection per operation. The engine keeps a pool of connections alive and
reuses them — the seed of what pgBouncer will do at scale (Phase 4).

### `app/db/models.py`
The `document_chunks` table — the same schema as `sql/001` — described as a
Python class. That is what an ORM is: the table is code, so queries are
Python too.

- `embedding` uses `pgvector`'s `Vector(768)` column type, which also
  provides `.cosine_distance()` — the Python equivalent of the raw
  `embedding <=> %s::vector` SQL the scripts wrote by hand.
- The attribute is `metadata_` because `metadata` is a reserved name in
  SQLAlchemy; it maps to the `metadata` column regardless.

**Concept: ORM & schema-as-code.**

---

## The LLM client

### `app/llm/ollama.py`
One async client for both Ollama endpoints:

- `embed(texts)` → `/api/embed`, validates every vector is 768-dim
  (same check your scripts had).
- `chat(system, user)` → `/api/chat`, non-streaming, temperature from
  config.

Every call is wrapped as `breaker(retry(call))`:

1. **Timeout** — `httpx.Timeout` caps how long any call may take.
2. **Retry with exponential backoff** (inner) — transient blips are
   absorbed; a fully-retried failure counts as *one* breaker failure.
3. **Circuit breaker** (outer) — when Ollama is down, requests fail
   instantly instead of hanging or hammering.

**Concept: resilience layering.** Order matters: the breaker fails fast
when open; the retry only burns attempts while the circuit is closed.

---

## Services (the RAG pipeline, extracted)

These three files contain the logic from your verified scripts, unchanged
in behavior, now callable from anywhere.

### `app/services/retrieval.py`
`search_chunks()` — embeds the query, then asks Postgres to rank chunks by
cosine distance (`ORDER BY embedding <=> query`), returning
`(chunk, distance)` closest-first.

### `app/services/generation.py`
`answer_question()` — the heart of the system, ported from `ask.py`:

1. Retrieve top-k chunks.
2. **Distance gate**: keep only chunks with distance ≤ `MAX_DISTANCE`.
3. If nothing clears the ceiling → **refuse**. The LLM is never called.
   The refusal is *structural* (code), not a prompt instruction.
4. Otherwise build numbered CONTEXT blocks and call Qwen3 with the strict
   grounding system prompt (answer only from context, cite like `[1]`,
   quote exact figures, no guessing).

**Concept: grounded generation.** The distance ceiling turns "closest
match" into "relevant enough" — the single most important design decision
in the system.

### `app/services/ingestion.py`
`ingest_chunk()` — embeds one piece of text and stores it with its vector
and metadata. (Later phases turn this into an async pipeline; for now it
is the synchronous single-chunk version.)

---

## The API layer

### `app/schemas/` (`query.py`, `search.py`, `ingest.py`)
Pydantic models that define the request/response **contract**: what fields
exist, their types, their limits (`question` must be 1–2000 chars, `top_k`
between 1 and 50). Bad input is rejected with a clean `422` before any of
your code runs. These same models generate the `/docs` page.

**Concept: schema validation at the boundary.** Invalid input never
reaches business logic.

### `app/api/deps.py`
Shared FastAPI dependencies: `SessionDep` (a DB session per request) and
`OllamaDep` (the shared OllamaClient from `app.state`). Endpoints just
declare what they need; FastAPI injects it.

**Concept: dependency injection.** Makes endpoints testable — tests swap
in fakes via `dependency_overrides`.

### `app/api/v1/router.py`
The three endpoints, all under the `/v1` prefix (**API versioning** — a
future `/v2` can coexist):

- `POST /v1/query` — grounded answer with citations, or a refusal
  (`"refused": true`). Refusing is a normal 200 response: it is a
  successful outcome of the contract, not an error.
- `POST /v1/search` — semantic search only; the LLM is never called.
- `POST /v1/ingest` — embed and store one chunk.

**Concept: API versioning.** `/v1/` from day one, so a future `/v2/` can
coexist without breaking clients.

### `app/api/health.py`
- `/health/live` — "is the process up?" Checks nothing. Kubernetes
  restarts the pod if this fails.
- `/health/ready` — "can we serve traffic?" Verifies the DB. Traffic stops
  routing to a pod whose readiness fails, without restarting it.

**Concept: liveness vs readiness probes.**

### `app/main.py`
The **app factory**. The `lifespan` runs at startup (creates the one shared
`OllamaClient` with its circuit breaker) and at shutdown (closes it and the
DB engine cleanly). Also registers the request-ID middleware.

---

## Migrations

### `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`
Alembic is **schema versioning**: schema changes are versioned Python
scripts. Any database — your laptop, a teammate's, CI's — can be brought
from empty to current with `alembic upgrade head`, and rolled back with
`alembic downgrade`. `env.py` pulls the DB URL from the app's own config
(nothing duplicated) and points at the ORM metadata so future migrations
can be auto-generated from model changes.

### `alembic/versions/0001_create_document_chunks.py`
The first migration: enables the `vector` extension and creates
`document_chunks` — exactly your `sql/001` schema, now versioned and
repeatable on any machine.

**Concept: database migrations.** Schema changes are code: reviewable,
reversible, and reproducible everywhere.

---

## Packaging & infrastructure

### `Dockerfile`
Multi-stage build: a builder stage installs dependencies, the final image
copies only the installed packages plus app code (smaller image), and runs
as a non-root user.

### `.dockerignore`
Keeps the image lean and — critically — excludes `.env`, so secrets never
bake into the image.

### `docker-compose.yml`
Three services:
- **postgres** — pgvector image, unchanged from your original, with a
  healthcheck.
- **redis** — included now, used from Phase 2 onward (cache, rate limits,
  queues).
- **api** — built from the Dockerfile. Key details:
  - `POSTGRES_HOST: postgres` — inside the Docker network, the database is
    the service name `postgres`, not `localhost`.
  - `OLLAMA_BASE_URL: http://host.docker.internal:11434` — Ollama runs on
    your host machine; the container reaches it through
    `host.docker.internal` (`extra_hosts` makes this work on Linux too).
  - `depends_on` with `condition: service_healthy` — the API does not
    start until Postgres is actually ready, not just started.

**Concepts: containerization, service networking, health-gated startup.**

---

## Tests

| File | What it proves | Needs |
|---|---|---|
| `tests/test_resilience.py` | Retry recovers from transient failures and gives up after the last attempt; the breaker opens after N failures, fails fast while open, recovers via a successful probe, re-opens after a failed probe | Nothing — pure asyncio |
| `tests/test_health.py` | `/health/live` works; every response carries an `x-request-id`; empty questions are rejected with 422 before touching services | Nothing |
| `tests/test_search_synthetic.py` | pgvector ranks by meaning: three chunks with synthetic orthogonal vectors, a fake Ollama returns a fixed vector, the matching axis must rank first with distance ≈ 0. Self-cleaning (deletes only its own marker rows) | Postgres only — runs in CI |
| `tests/test_rag_local.py` | The full grounding contract with real Ollama: matched question → cited answer; unrelated question → refusal | Ollama + Postgres |

The synthetic-vector trick is the key idea: you do not need a real
embedding model to prove the database ranks by meaning — orthogonal unit
vectors make the math obvious (matching axis → distance 0).

**Concepts: test pyramid, mocking external services, self-cleaning
fixtures (the same marker pattern as your `verify_*` scripts).**

---

## CI

`.github/workflows/ci.yml` — two jobs on every push:
- **lint**: ruff check + format check + mypy
- **test**: spins up a pgvector service container, runs
  `pytest -m "not local"` — everything except the Ollama tests, which stay
  local because downloading models in CI is impractical.

**Concept: continuous integration.** Broken code cannot reach main
quietly; the build turns red.

---

## Support files

| File | Purpose |
|---|---|
| `requirements.txt` / `requirements-dev.txt` | Runtime vs dev dependencies — production installs stay lean |
| `pytest.ini` | Test config: asyncio mode, markers, `pythonpath` |
| `mypy.ini` | Type-check config (lenient, skips migrations/scripts/tests) |
| `.env.example` | Template of every config variable, with comments |
| `scripts/try_api.ps1` | Manual API testing helper (health/ingest/query/search) |
| `scripts/verify_*.py`, `ask.py`, ... | Your original stage scripts — untouched, still work |

---

## The one-paragraph summary

Phase 0 took the proven RAG pipeline out of `ask.py` and gave it an
architecture: a **contract** (schemas), a **brain** (services), a **voice**
(Ollama client with timeout/retry/breaker), **memory** (SQLAlchemy +
pgvector + Alembic), **nervous system** (structured logs with request IDs),
**health checks** (live/ready), a **container** (Docker), and a **robot
gatekeeper** (CI). Same brain, new body — ready for Phase 1 to add hybrid
retrieval, reranking, and the evaluation harness.