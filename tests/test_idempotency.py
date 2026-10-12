"""Phase 2, step 3: idempotency keys on POST /v1/ingest.

What these tests prove:
  - a retry with the same key stores NOTHING a second time and returns the
    first response; different keys / no key behave as before
  - 12 simultaneous requests with one key run the work exactly once
  - the same key with a different body is rejected, not silently answered
  - keys are private to the caller that made them
  - a failed or cancelled first attempt releases the key, so the retry runs
  - a crashed server can't block a key forever (the claim expires)
  - a slow request can't delete the claim of whoever took the key after it
  - Redis down + a key -> 503 and no work done; no key -> unaffected

Needs a real Redis (docker compose locally, service container in CI). The
database and Ollama are not needed: the storage step is replaced by a fake
that counts how often it really ran.
"""

import asyncio
import uuid

import pytest
from redis.asyncio import Redis

from app.api.v1 import router as router_module
from app.core.config import Settings, get_settings
from app.core.idempotency import (
    IdempotencyInProgress,
    IdempotencyMismatch,
    IdempotencyStore,
    IdempotencyUnavailable,
    Outcome,
    create_idempotency_store,
    fingerprint,
    run_once,
)
from app.main import app
from tests.auth_helpers import bearer, make_token

pytestmark = pytest.mark.redis


# ----------------------------------------------------------------------- helpers


def _settings(**overrides) -> Settings:
    base = get_settings().model_dump()
    base.update(idempotency_key_prefix=f"test-idem-unit:{uuid.uuid4().hex}")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def make_store():
    created: list[IdempotencyStore] = []

    def _make(**overrides) -> IdempotencyStore:
        store = create_idempotency_store(_settings(**overrides))
        created.append(store)
        return store

    yield _make
    for store in created:
        await store.close()


class FakeIngest:
    """Replaces ingest_chunk(): counts real executions, optionally slow/failing."""

    def __init__(self) -> None:
        self.calls = 0
        self.delay = 0.0
        self.fail_next = False

    async def __call__(self, session, ollama, content, metadata=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("embedding service exploded")
        return type("Chunk", (), {"id": 1000 + self.calls})()


@pytest.fixture
def fake_ingest(monkeypatch):
    fake = FakeIngest()
    monkeypatch.setattr(router_module, "ingest_chunk", fake)
    return fake


def _key() -> str:
    return f"key-{uuid.uuid4().hex[:12]}"


async def _post(client, key, content="hydraulic filter note", headers=None, **extra):
    h = dict(headers or {})
    if key is not None:
        # HTTP header values are bytes on the wire (Latin-1); sending bytes
        # also lets the test send characters httpx would refuse as text.
        h["Idempotency-Key"] = key.encode("latin-1")
    return await client.post(
        "/v1/ingest", json={"content": content, **extra}, headers=h
    )


# ------------------------------------------------------------------ over HTTP


async def test_retry_with_same_key_stores_once_and_replays(client, fake_ingest):
    key = _key()
    first = await _post(client, key)
    second = await _post(client, key)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()  # same chunk id, not a new one
    assert fake_ingest.calls == 1
    assert "idempotent-replay" not in first.headers
    assert second.headers["idempotent-replay"] == "true"


async def test_without_a_key_every_request_runs(client, fake_ingest):
    a = await _post(client, None)
    b = await _post(client, None)
    assert a.json()["id"] != b.json()["id"]
    assert fake_ingest.calls == 2


async def test_different_keys_are_independent(client, fake_ingest):
    await _post(client, _key())
    await _post(client, _key())
    assert fake_ingest.calls == 2


async def test_same_key_different_body_is_rejected(client, fake_ingest):
    key = _key()
    await _post(client, key, content="original text")
    response = await _post(client, key, content="a different text")

    assert response.status_code == 422
    assert "different request body" in response.json()["detail"]
    assert fake_ingest.calls == 1  # the second body was NOT stored


async def test_metadata_is_part_of_the_fingerprint(client, fake_ingest):
    key = _key()
    await _post(client, key, metadata={"source": "a"})
    response = await _post(client, key, metadata={"source": "b"})
    assert response.status_code == 422


async def test_key_order_in_the_body_does_not_matter(client, fake_ingest):
    key = _key()
    await client.post(
        "/v1/ingest",
        json={"content": "x", "metadata": {"a": 1, "b": 2}},
        headers={"Idempotency-Key": key},
    )
    again = await client.post(
        "/v1/ingest",
        json={"metadata": {"b": 2, "a": 1}, "content": "x"},
        headers={"Idempotency-Key": key},
    )
    assert again.status_code == 200
    assert again.headers["idempotent-replay"] == "true"
    assert fake_ingest.calls == 1


async def test_concurrent_requests_with_one_key_run_exactly_once(client, fake_ingest):
    fake_ingest.delay = 0.3  # keep the first request in flight while others arrive
    key = _key()

    responses = await asyncio.gather(*[_post(client, key) for _ in range(12)])

    statuses = [r.status_code for r in responses]
    assert fake_ingest.calls == 1, "the work ran more than once"
    assert statuses.count(200) >= 1
    assert set(statuses) <= {200, 409}  # winner (+ replays) and in-progress
    for r in responses:
        if r.status_code == 409:
            assert r.headers["retry-after"] == "1"
    # Everyone who got an answer got the SAME answer.
    assert len({r.json()["id"] for r in responses if r.status_code == 200}) == 1


async def test_retry_during_first_request_gets_409_then_replay(client, fake_ingest):
    fake_ingest.delay = 0.4
    key = _key()

    first = asyncio.create_task(_post(client, key))
    await asyncio.sleep(0.1)  # first request is mid-flight
    during = await _post(client, key)
    done = await first
    after = await _post(client, key)

    assert during.status_code == 409
    assert done.status_code == 200
    assert after.status_code == 200
    assert after.headers["idempotent-replay"] == "true"
    assert after.json() == done.json()
    assert fake_ingest.calls == 1


async def test_keys_are_private_to_the_caller(anon_client, fake_ingest):
    key = _key()
    alice = bearer(make_token("alice", ["ingest"]))
    bob = bearer(make_token("bob", ["ingest"]))

    a = await _post(anon_client, key, headers=alice)
    b = await _post(anon_client, key, headers=bob)

    # Same key string, different callers: two real inserts, no cross-talk.
    assert a.json()["id"] != b.json()["id"]
    assert "idempotent-replay" not in b.headers
    assert fake_ingest.calls == 2


async def test_failed_first_attempt_releases_the_key(client, fake_ingest):
    key = _key()
    fake_ingest.fail_next = True
    with pytest.raises(RuntimeError):  # ASGI test transport re-raises server errors
        await _post(client, key)

    retry = await _post(client, key)
    assert retry.status_code == 200  # not stuck as "in progress"
    assert "idempotent-replay" not in retry.headers
    assert fake_ingest.calls == 2


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "semi;colon", "naïve", ""])
async def test_malformed_keys_are_rejected(client, fake_ingest, bad):
    response = await _post(client, bad)
    assert response.status_code == 400
    assert fake_ingest.calls == 0


async def test_redis_down_with_a_key_is_503_and_does_nothing(client, fake_ingest):
    dead = IdempotencyStore(
        Redis.from_url(
            "redis://127.0.0.1:1/0", socket_timeout=0.2, socket_connect_timeout=0.2
        ),
        _settings(),
    )
    original = app.state.idempotency
    app.state.idempotency = dead
    try:
        with_key = await _post(client, _key())
        without_key = await _post(client, None)
    finally:
        app.state.idempotency = original
        await dead.close()

    assert with_key.status_code == 503
    assert with_key.headers["retry-after"] == "5"
    assert fake_ingest.calls == 1  # only the keyless request ran
    assert without_key.status_code == 200


async def test_idempotency_can_be_disabled(client, fake_ingest):
    original = app.state.idempotency
    app.state.idempotency = None  # what IDEMPOTENCY_ENABLED=false produces
    try:
        key = _key()
        await _post(client, key)
        await _post(client, key)
    finally:
        app.state.idempotency = original
    assert fake_ingest.calls == 2  # header ignored


async def test_scope_still_applies(anon_client, fake_ingest):
    reader = bearer(make_token("reader-only", ["query"]))
    response = await _post(anon_client, _key(), headers=reader)
    assert response.status_code == 403
    assert fake_ingest.calls == 0


# ------------------------------------------------------------------ the store


async def test_begin_outcomes_walk_the_whole_lifecycle(make_store):
    store = make_store()
    fp = fingerprint({"content": "x"})

    first = await store.begin("alice", "k1", fp)
    assert first.outcome is Outcome.CLAIMED and first.token

    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.IN_PROGRESS
    assert (await store.begin("alice", "k1", "other")).outcome is Outcome.MISMATCH

    await store.complete("alice", "k1", fp, {"id": 7})
    replay = await store.begin("alice", "k1", fp)
    assert replay.outcome is Outcome.REPLAY and replay.body == {"id": 7}
    assert (await store.begin("alice", "k1", "other")).outcome is Outcome.MISMATCH


async def test_done_entries_expire_after_the_ttl_setting(make_store):
    store = make_store(idempotency_ttl_seconds=90)
    fp = fingerprint({"a": 1})
    await store.begin("alice", "k1", fp)
    await store.complete("alice", "k1", fp, {"id": 1})

    ttl = await store.redis.ttl(store._redis_key("alice", "k1"))
    assert 80 <= ttl <= 90


async def test_abort_releases_only_our_own_claim(make_store):
    store = make_store()
    fp = fingerprint({"a": 1})
    mine = await store.begin("alice", "k1", fp)

    await store.abort("alice", "k1", "someone-elses-token")  # must be a no-op
    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.IN_PROGRESS

    await store.abort("alice", "k1", mine.token)  # the real owner
    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.CLAIMED


async def test_abort_cannot_delete_a_finished_result(make_store):
    store = make_store()
    fp = fingerprint({"a": 1})
    mine = await store.begin("alice", "k1", fp)
    await store.complete("alice", "k1", fp, {"id": 5})

    await store.abort("alice", "k1", mine.token)  # late, stale abort
    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.REPLAY


async def test_a_crashed_claim_expires_instead_of_blocking_forever(make_store):
    store = make_store(idempotency_lock_seconds=1)
    fp = fingerprint({"a": 1})
    await store.begin("alice", "k1", fp)  # ...and the server dies here

    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.IN_PROGRESS
    await asyncio.sleep(1.3)
    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.CLAIMED


async def test_callers_with_colons_cannot_collide(make_store):
    store = make_store()
    assert store._redis_key("a:b", "c") != store._redis_key("a", "b:c")


async def test_unavailable_store_raises(make_store):
    dead = IdempotencyStore(
        Redis.from_url(
            "redis://127.0.0.1:1/0", socket_timeout=0.2, socket_connect_timeout=0.2
        ),
        _settings(),
    )
    try:
        with pytest.raises(IdempotencyUnavailable):
            await dead.begin("alice", "k1", "fp")
    finally:
        await dead.close()


# -------------------------------------------------------------------- run_once


async def test_run_once_releases_the_key_on_error(make_store):
    store = make_store()
    fp = fingerprint({"a": 1})

    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await run_once(store, "alice", "k1", fp, boom)

    ran = []

    async def ok():
        ran.append(1)
        return {"id": 1}

    result = await run_once(store, "alice", "k1", fp, ok)
    assert ran and not result.replayed


async def test_run_once_releases_the_key_when_cancelled(make_store):
    # A client that hangs up cancels the request task mid-work.
    store = make_store()
    fp = fingerprint({"a": 1})
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)
        return {"id": 1}

    task = asyncio.create_task(run_once(store, "alice", "k1", fp, slow))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await store.begin("alice", "k1", fp)).outcome is Outcome.CLAIMED


async def test_run_once_raises_the_right_exceptions(make_store):
    store = make_store()
    fp = fingerprint({"a": 1})
    gate = asyncio.Event()

    async def slow():
        await gate.wait()
        return {"id": 1}

    first = asyncio.create_task(run_once(store, "alice", "k1", fp, slow))
    await asyncio.sleep(0.05)

    async def never():
        raise AssertionError("must not run")

    with pytest.raises(IdempotencyInProgress):
        await run_once(store, "alice", "k1", fp, never)
    with pytest.raises(IdempotencyMismatch):
        await run_once(store, "alice", "k1", "different", never)

    gate.set()
    assert (await first).replayed is False
    assert (await run_once(store, "alice", "k1", fp, never)).replayed is True


async def test_release_survives_repeated_cancellation(make_store):
    # Starlette/anyio cancel scopes are level-triggered: once cancelled, EVERY
    # await inside the scope raises again. A plain `await store.abort(...)` in
    # the cleanup path would be cancelled before it reaches Redis and the key
    # would stay "in progress" until its claim expired. The shield lets the
    # release finish in the background.
    import anyio

    class SlowReleaseStore(IdempotencyStore):
        async def abort(self, subject, key, token):
            # Any cancellation that reaches this await skips the release.
            await asyncio.sleep(0.05)
            await super().abort(subject, key, token)

    base = make_store()
    store = SlowReleaseStore(base.redis, base.settings)
    fp = fingerprint({"a": 1})
    started = asyncio.Event()
    scope_holder: dict = {}

    async def slow():
        started.set()
        await asyncio.sleep(30)
        return {"id": 1}

    async def request():
        with anyio.CancelScope() as scope:
            scope_holder["scope"] = scope
            await run_once(store, "alice", "k1", fp, slow)

    task = asyncio.create_task(request())
    await started.wait()
    scope_holder["scope"].cancel()
    await task
    await asyncio.sleep(0.3)  # the shielded release finishes in the background

    assert (await base.begin("alice", "k1", fp)).outcome is Outcome.CLAIMED
