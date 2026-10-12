"""Idempotency keys for non-repeatable requests (Phase 2, step 3).

The problem: `POST /v1/ingest` creates a row. If the client's connection drops
after the server inserted the chunk but before the client saw the answer, the
client retries -- and now the corpus holds the chunk twice. Duplicates skew
retrieval (the same text outranks everything near it) and nothing flags them.

The fix: the client attaches a unique `Idempotency-Key` header. The first
request with that key does the work and the response is remembered; a retry
with the same key gets the remembered response back and does NOT run again.

How a key moves through Redis (one entry per caller+key):

    (absent) --begin--> in_progress --complete--> done     (kept for 24 h)
                             |
                             +--abort (error/cancel)--> (absent)  so a retry can run

  - `begin` claims the key with SET NX, which is atomic: of any number of
    simultaneous requests with the same key, exactly one wins the claim.
  - The losers see either `in_progress` (the winner is still working: 409) or
    `done` (replay the stored response).
  - `in_progress` has a short expiry, so a server that dies mid-request can't
    block the key forever.
  - Each claim carries a random token, so a slow request whose claim expired
    cannot delete the claim of whoever took the key after it.

Each key is also bound to a fingerprint of the request body. Re-using a key
with a *different* body is a client bug (or an attack), and silently returning
the first request's answer would hide it, so it is rejected.

Redis down: callers that sent a key get an error (the HTTP layer maps it to
503) instead of silently running without protection -- sending a key is
asking for exactly-once, and "probably once" would break that promise.
"""

import asyncio
import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class IdempotencyMismatch(Exception):
    """The key was already used with a different request body."""


class IdempotencyInProgress(Exception):
    """The first request with this key is still running."""


class IdempotencyUnavailable(Exception):
    """The idempotency store (Redis) can't be reached."""


class Outcome(Enum):
    CLAIMED = "claimed"  # we own the key: go do the work
    REPLAY = "replay"  # finished before: return the stored response
    IN_PROGRESS = "in_progress"  # someone else is doing it right now
    MISMATCH = "mismatch"  # key reused with a different body


@dataclass(frozen=True)
class BeginResult:
    outcome: Outcome
    token: str | None = None  # proof of ownership; set when CLAIMED
    body: dict | None = None  # stored response; set when REPLAY


@dataclass(frozen=True)
class IdempotentResult:
    body: dict
    replayed: bool


# Release a claim only if it is still OUR in-progress claim (token matches).
# GET-then-DEL as separate calls could delete a newer owner's claim.
_RELEASE_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local entry = cjson.decode(raw)
if entry.state == 'in_progress' and entry.token == ARGV[1] then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""


def fingerprint(payload: dict) -> str:
    """Stable hash of a request body (key order and whitespace don't matter)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyStore:
    def __init__(self, redis: Redis, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = redis
        self._release = redis.register_script(_RELEASE_LUA)

    def _redis_key(self, subject: str, key: str) -> str:
        # Hash (caller, key) together. Plain concatenation would let caller
        # "a:b" + key "c" collide with caller "a" + key "b:c" -- two different
        # callers sharing one entry.
        digest = hashlib.sha256(f"{subject}\0{key}".encode()).hexdigest()
        return f"{self.settings.idempotency_key_prefix}:{digest}"

    async def begin(self, subject: str, key: str, request_fp: str) -> BeginResult:
        rkey = self._redis_key(subject, key)
        token = uuid.uuid4().hex
        claim = json.dumps(
            {"state": "in_progress", "fingerprint": request_fp, "token": token}
        )
        try:
            # A key can expire between a failed claim and the read that
            # follows it; one more attempt covers that gap.
            for _ in range(2):
                claimed = await self.redis.set(
                    rkey, claim, nx=True, ex=self.settings.idempotency_lock_seconds
                )
                if claimed:
                    return BeginResult(Outcome.CLAIMED, token=token)
                raw = await self.redis.get(rkey)
                if raw is None:
                    continue
                entry = json.loads(raw)
                if entry["fingerprint"] != request_fp:
                    return BeginResult(Outcome.MISMATCH)
                if entry["state"] == "done":
                    return BeginResult(Outcome.REPLAY, body=entry["body"])
                return BeginResult(Outcome.IN_PROGRESS)
        except (RedisError, OSError) as exc:
            raise IdempotencyUnavailable(str(exc)) from exc
        raise IdempotencyUnavailable("could not settle the key's state")

    async def complete(
        self, subject: str, key: str, request_fp: str, body: dict
    ) -> None:
        """Remember the response. Best effort: the work is already done."""
        entry = json.dumps({"state": "done", "fingerprint": request_fp, "body": body})
        try:
            await self.redis.set(
                self._redis_key(subject, key),
                entry,
                ex=self.settings.idempotency_ttl_seconds,
            )
        except (RedisError, OSError) as exc:
            # Nothing to undo -- the row exists. The key stays in_progress
            # until its short expiry; log loudly so the gap is visible.
            logger.error("idempotency: could not store response", error=str(exc))

    async def abort(self, subject: str, key: str, token: str) -> None:
        """Release our claim after a failure so the client can retry."""
        try:
            await self._release(keys=[self._redis_key(subject, key)], args=[token])
        except (RedisError, OSError) as exc:
            logger.error("idempotency: could not release key", error=str(exc))

    async def close(self) -> None:
        await self.redis.aclose()


async def run_once(
    store: IdempotencyStore,
    subject: str,
    key: str,
    request_fp: str,
    action: Callable[[], Awaitable[dict]],
) -> IdempotentResult:
    """Run `action` at most once per (subject, key); replay its result after.

    `action` returns the JSON-able response body.
    """
    begun = await store.begin(subject, key, request_fp)
    if begun.outcome is Outcome.MISMATCH:
        raise IdempotencyMismatch
    if begun.outcome is Outcome.IN_PROGRESS:
        raise IdempotencyInProgress
    if begun.outcome is Outcome.REPLAY:
        assert begun.body is not None
        return IdempotentResult(begun.body, replayed=True)

    assert begun.token is not None
    try:
        body = await action()
    except BaseException:
        # Includes cancellation (client hung up). Shielded so the release
        # still completes while the task is being torn down.
        await asyncio.shield(store.abort(subject, key, begun.token))
        raise
    await asyncio.shield(store.complete(subject, key, request_fp, body))
    return IdempotentResult(body, replayed=False)


def create_idempotency_store(settings: Settings | None = None) -> IdempotencyStore:
    settings = settings or get_settings()
    redis = Redis.from_url(
        settings.redis_url, socket_timeout=0.5, socket_connect_timeout=0.5
    )
    return IdempotencyStore(redis, settings)
