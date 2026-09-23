"""Phase 2, step 1: Redis token-bucket rate limiting.

What these tests prove:
  - a bucket allows exactly `capacity` requests, then rejects with a
    Retry-After that is actually long enough
  - tokens come back over time (refill), and clients don't share buckets
  - the Lua script is atomic: a burst of concurrent requests can't sneak
    past the limit (the race a GET-then-SET version would lose)
  - over HTTP: /v1/* returns 429 + Retry-After when the bucket is empty,
    while /health/* is never limited
  - Redis down: fail-open lets requests through, fail-closed rejects them

Tests marked `redis` need a real Redis (docker compose provides one on
localhost:6379; CI runs a Redis service container). Every test uses its own
key prefix, so runs never interfere with each other or with a live API
sharing the same Redis.
"""

import asyncio
import uuid

import pytest
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.core.rate_limit import RateLimiter, create_rate_limiter
from app.main import app


def _settings(**overrides) -> Settings:
    base = get_settings().model_dump()
    base.update(rate_limit_key_prefix=f"test-ratelimit:{uuid.uuid4().hex}")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def make_limiter():
    """Build limiters against the real Redis; close them all afterwards."""
    created: list[RateLimiter] = []

    def _make(**overrides) -> RateLimiter:
        limiter = create_rate_limiter(_settings(**overrides))
        created.append(limiter)
        return limiter

    yield _make

    for limiter in created:
        await limiter.close()


# ---------------------------------------------------------------- bucket logic


@pytest.mark.redis
async def test_allows_capacity_then_rejects(make_limiter):
    limiter = make_limiter(rate_limit_capacity=3, rate_limit_refill_per_second=0.1)

    results = [await limiter.hit("client-a") for _ in range(4)]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results[:3]] == [2, 1, 0]
    rejected = results[3]
    assert rejected.degraded is False
    # empty bucket at 0.1 tokens/s -> 10s until the next token
    assert 1 <= rejected.retry_after_seconds <= 10


@pytest.mark.redis
async def test_tokens_refill_over_time(make_limiter):
    limiter = make_limiter(rate_limit_capacity=1, rate_limit_refill_per_second=20)

    assert (await limiter.hit("client-a")).allowed
    assert not (await limiter.hit("client-a")).allowed

    await asyncio.sleep(0.1)  # 20 tokens/s * 0.1s = 2 tokens, capped at 1
    assert (await limiter.hit("client-a")).allowed


@pytest.mark.redis
async def test_clients_have_separate_buckets(make_limiter):
    limiter = make_limiter(rate_limit_capacity=1, rate_limit_refill_per_second=0.1)

    assert (await limiter.hit("client-a")).allowed
    assert not (await limiter.hit("client-a")).allowed
    assert (await limiter.hit("client-b")).allowed  # unaffected by client-a


@pytest.mark.redis
async def test_concurrent_burst_cannot_exceed_capacity(make_limiter):
    # The reason the bucket is a Lua script: 25 requests racing for 5
    # tokens must yield exactly 5 allowed, not "5-ish".
    limiter = make_limiter(rate_limit_capacity=5, rate_limit_refill_per_second=0.01)

    results = await asyncio.gather(*(limiter.hit("burst") for _ in range(25)))

    assert sum(r.allowed for r in results) == 5


@pytest.mark.redis
async def test_idle_bucket_key_expires(make_limiter):
    # Buckets must not pile up in Redis forever: a full bucket is the same
    # as no bucket, so the key carries a TTL of ~capacity/refill seconds.
    limiter = make_limiter(rate_limit_capacity=10, rate_limit_refill_per_second=1)
    await limiter.hit("client-a")

    ttl = await limiter.redis.ttl(f"{limiter.key_prefix}:client-a")
    assert 0 < ttl <= 11


# ------------------------------------------------------------ Redis unavailable


def _limiter_with_dead_redis(fail_open: bool) -> RateLimiter:
    # Port 1: nothing listens there, so every call fails immediately.
    dead = Redis.from_url(
        "redis://127.0.0.1:1/0", socket_timeout=0.2, socket_connect_timeout=0.2
    )
    return RateLimiter(dead, _settings(rate_limit_fail_open=fail_open))


async def test_redis_down_fail_open_allows():
    limiter = _limiter_with_dead_redis(fail_open=True)
    try:
        # enough calls to trip the breaker, so both the "Redis error" and
        # the "circuit open, Redis not even tried" paths are covered
        results = [await limiter.hit("client-a") for _ in range(5)]
    finally:
        await limiter.close()

    assert all(r.allowed and r.degraded for r in results)
    assert limiter.breaker.state == "open"


async def test_redis_down_fail_closed_rejects():
    limiter = _limiter_with_dead_redis(fail_open=False)
    try:
        result = await limiter.hit("client-a")
    finally:
        await limiter.close()

    assert result.allowed is False
    assert result.degraded is True


# --------------------------------------------------------------------- over HTTP


@pytest.fixture
async def limited_client(client, make_limiter):
    """The app with a tiny 2-request bucket swapped in."""
    app_state = app.state
    original = app_state.rate_limiter
    app_state.rate_limiter = make_limiter(
        rate_limit_capacity=2, rate_limit_refill_per_second=0.1
    )
    yield client
    app_state.rate_limiter = original


@pytest.mark.redis
async def test_v1_returns_429_with_retry_after_when_bucket_empty(limited_client):
    # Invalid bodies (422) still spend a token: the limiter runs before the
    # endpoint, so no Ollama/Postgres is needed to exhaust the bucket.
    for _ in range(2):
        response = await limited_client.post("/v1/query", json={"question": ""})
        assert response.status_code == 422

    response = await limited_client.post("/v1/query", json={"question": ""})

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "0"
    assert response.headers["x-request-id"]  # still traceable in the logs


@pytest.mark.redis
async def test_bucket_is_shared_across_v1_endpoints(limited_client):
    await limited_client.post("/v1/query", json={"question": ""})
    await limited_client.post("/v1/search", json={"query": ""})

    response = await limited_client.post("/v1/ingest", json={"content": ""})
    assert response.status_code == 429


@pytest.mark.redis
async def test_health_is_never_rate_limited(limited_client):
    for _ in range(5):
        await limited_client.post("/v1/query", json={"question": ""})

    for _ in range(5):
        assert (await limited_client.get("/health/live")).status_code == 200


async def test_rate_limiting_can_be_disabled(client):
    app_state = app.state
    original = app_state.rate_limiter
    app_state.rate_limiter = None  # what RATE_LIMIT_ENABLED=false produces
    try:
        for _ in range(5):
            response = await client.post("/v1/query", json={"question": ""})
            assert response.status_code == 422
            assert "x-ratelimit-limit" not in response.headers
    finally:
        app_state.rate_limiter = original


@pytest.mark.redis
async def test_successful_response_carries_rate_limit_headers(
    limited_client, monkeypatch
):
    # A real 200 (retrieval faked out, as in test_pipeline_wiring) must tell
    # the client its budget, so well-behaved callers can slow down on their
    # own before ever seeing a 429.
    from app.api.v1 import router as router_module

    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        return []

    monkeypatch.setattr(router_module, "retrieve", fake_retrieve)

    response = await limited_client.post("/v1/search", json={"query": "hydraulics"})

    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "1"
