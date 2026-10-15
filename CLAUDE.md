# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AeroIntel is an aviation-maintenance RAG service, built stage by stage with each
stage proven by its own tests before the next starts. Phase 1 (hybrid
retrieval + eval harness) is closed. Phase 2 is in progress: Redis rate limiting,
JWT auth (scopes `query`/`ingest`), idempotency keys on `/v1/ingest` and SSE
streaming (`/v1/query/stream`) are done; the semantic cache is not started. PDF
ingestion with page citations (a Phase 4 item) was pulled forward and exists. Nothing
else from Phase 3 on is built (event-driven ingestion, k8s deploy, observability).
See the Status table and Roadmap in `README.md` for the full plan — don't reimplement
later-phase work prematurely.

**Non-negotiable rule:** the model never answers from its own knowledge. It only
sees numbered context blocks pulled from the database via `POST /v1/query`, and
if nothing is close enough (distance > `MAX_DISTANCE`), the LLM is never called —
refusal is structural, enforced in `app/services/generation.py`, not a prompt
instruction.

## Commands

```powershell
# Setup (one-time)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env   # then set POSTGRES_PASSWORD

# Run the stack
docker compose up -d --build
alembic upgrade head                    # apply schema migrations (from host, first time)
python scripts/load_sample_data.py      # optional: load the 6 synthetic chunks

# Tests
pytest -m "not local" -v   # unit + API contract + pgvector ranking — what CI runs
pytest -m local            # full RAG tests, need real Ollama (~30s)
pytest tests/test_hybrid.py -v          # run a single test file
pytest tests/test_hybrid.py::test_name  # run a single test

# Lint / format / typecheck
ruff check app tests
ruff format app tests
mypy app                   # note: mypy excludes alembic/versions, scripts, tests

# Retrieval eval harness (needs Postgres + sample data; add --retriever/--reranker/--faithfulness to A/B test)
python -m evals.run_eval
python -m evals.run_eval --retriever hybrid --reranker crossencoder --faithfulness

# Embedding version migration (rows are tagged with embedding_model; never silently mix models)
python scripts/backfill_embeddings.py --dry-run
```

Use `curl.exe`, not `curl`, in PowerShell — bare `curl` is an `Invoke-WebRequest`
alias with different flags. CI (`.github/workflows/ci.yml`) runs ruff + mypy +
`pytest -m "not local"` against a real pgvector service container on every push.

## Architecture

Request flow for `POST /v1/query`:

```
embed question (EmbeddingGemma via Ollama)
  -> app/services/pipeline.py:retrieve() — vector or hybrid search, then optional rerank
  -> distance <= MAX_DISTANCE? no -> refuse, LLM never called
  -> yes -> build numbered CONTEXT blocks, Qwen3 generates, forced to cite [n]
```

`app/services/pipeline.py:retrieve()` is the single retrieval entrypoint used by
*both* the API and the eval harness — this is deliberate, so what's measured is
exactly what's served. It composes, all independently feature-flagged and off by
default:
- `app/services/retrieval.py` — plain pgvector cosine search
- `app/services/hybrid.py` — BM25 + pgvector merged via Reciprocal Rank Fusion (`HYBRID_ENABLED`)
- `app/services/reranking.py` — cross-encoder rescoring of top candidates (`RERANKER=crossencoder`, needs `requirements-rerank.txt`)

Layering: `api/v1/router.py` (HTTP) -> `services/*.py` (RAG logic, no HTTP
concerns) -> `db/` (async SQLAlchemy 2.0 + pgvector) and `llm/ollama.py`
(`OllamaClient`, wraps `embed()`/`chat()`). Every Ollama call goes through
`app/core/resilience.py` (timeout -> retry-with-backoff -> circuit breaker), so
a slow/dead Ollama fails fast instead of hanging the API. `app/core/config.py`
holds typed `Settings` from env/`.env` with no defaults for secrets;
`app/core/logging.py` provides structlog JSON logs + request-ID middleware.
`app/core/rate_limit.py` is a Redis token bucket (atomic Lua script) applied to
every `/v1` route via the router-level `enforce_rate_limit` dependency in
`app/api/deps.py`; `/health/*` is deliberately outside it. A Redis outage is
handled by a circuit breaker plus `RATE_LIMIT_FAIL_OPEN` (default: allow).

Auth: `app/core/auth.py` holds JWT issue/verify (one `decode_access_token`
function, so Keycloak can replace it in Phase 7) and `validate_auth_settings`,
called from the API lifespan — not from `Settings` — so alembic/scripts need no
JWT secret. The router-level `authenticate` dependency runs *before*
`enforce_rate_limit`, so buckets are keyed by the token subject (`/auth/token`,
which has no token yet, is limited by IP). Routes declare `require_scope(...)`.

Idempotency: `app/core/idempotency.py` makes `POST /v1/ingest` safe to retry
when the client sends an `Idempotency-Key` header. A Redis `SET NX` claims the
key atomically (`in_progress` -> `done`); entries are namespaced per caller and
bound to a fingerprint of the request body; a failed or cancelled request
releases its claim (token-checked, shielded from cancellation). Unlike rate
limiting this **fails closed**: a request that sent a key gets 503 if Redis is
down, because a key means "exactly once". Requests without a key are unaffected.

Streaming: `POST /v1/query/stream` calls `services/generation.py:prepare_answer()`
(retrieval + distance gate + prompt, DB session committed) *before* the response
starts, then relays `OllamaClient.chat_stream()` as SSE events
(`meta` -> `token`* -> `done`, or `error` in-band). `answer_question()` uses the
same `prepare_answer()`, so the plain and streaming endpoints cannot drift. A
refused query is `meta` + `done` and `stream_answer()` raises if asked to stream
one -- the model is unreachable on refusal. Only *opening* the Ollama stream gets
retry/breaker; mid-stream failures are not retried (it would duplicate text).
Don't use the DB session inside the streaming generator.

Documents: `POST /v1/documents` and `scripts/ingest_pdfs.py` both call
`services/documents.py:ingest_pdf()`: SHA-256 dedup lookup (then commit, to release
the pooled connection), `services/pdf_text.py` (pypdf text, repeated header/footer
removal, printed page labels, sentence-bound chunks that never cross a page), optional
`services/ocr.py` for blank pages, batched embedding, then one transaction writing the
`documents` row and its chunks (`document_id`, `chunk_index`, `page`, `page_label`).
Never hold a DB connection across reading/embedding. The new tables/columns live in
migration 0004, which is idempotent on purpose (the integration tests' `create_all`
can create `documents` first). Chunks from plain `/v1/ingest` have NULL document
fields and still work.

Tests: `tests/auth_helpers.py` configures auth env (and per-process Redis key
prefixes, so runs never interfere) before `app` loads; `conftest.py` provides an
authenticated `client` (admin: query + ingest) and `anon_client` (no credentials).

**Prompts are versioned files**, not string literals: `app/prompts/<name>.<version>.txt`.
Changing wording means adding a new `vN.txt` file, not editing one in place —
each prompt's sha256 is logged/recorded, so a silent edit is detectable and old
eval numbers don't get silently misattributed. `PROMPT_VERSION` env var pins a
version; unset uses the newest file.

**Embeddings are version-tracked**: every `document_chunks` row records the
model that produced its vector (`embedding_model`). Vectors from different
models are never comparable — a model swap goes through
`scripts/backfill_embeddings.py`, not a direct `EMBEDDING_MODEL` flip.

## Data

The database holds 6 hand-written synthetic chunks
(`scripts/load_sample_data.py`) — explicitly placeholder content, not real
aircraft manual text — plus whatever PDFs have been ingested (PDF ingestion with
page citations exists; collections, HNSW and background ingestion do not yet).
Don't treat retrieval-quality numbers as meaningful beyond the toy corpus until a
golden question set exists for real documents. `data/` (downloaded PDFs) is gitignored;
`scripts/fetch_sample_docs.ps1` fetches free FAA manuals (public domain).

## Conventions

- `tests/` and `evals/` are excluded from mypy but linted by ruff.
- Pytest markers: unmarked = fast/CI-safe, `integration` = needs real Postgres,
  `redis` = needs real Redis (compose on localhost:6379 / CI service container),
  `local` = needs real Ollama + Postgres. Don't add `local`-marked tests to CI.
- `scripts/` holds the original per-stage verification scripts (`ask.py`,
  `verify_*.py`) predating the FastAPI service — they still work standalone and
  are useful for manual/local checks, not part of the pytest suite.
- Before/after any retrieval change, compare `evals/run_eval` numbers — don't
  judge retrieval quality by eyeballing a few queries.
