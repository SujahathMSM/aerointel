"""Semantic answer cache (Phase 2, step 5).

The expensive part of a query is the model writing the answer: seconds on a
local GPU, a paid call on the Anthropic API. Retrieval and the gates are
cheap. So the cache replaces only that one call, and only when it is safe to:

  1. Every gate has already run for the *new* question (distance, unknown
     words, answer check). A refused question never reaches the cache, so the
     cache cannot turn a refusal into an answer.
  2. The context the model would read is *identical*: same passages, same
     order (the [n] citations point at them), same system prompt, same model
     and generation settings. All of that is hashed into the bucket key by the
     caller, so new documents, re-chunking, a prompt or model change simply
     land in a new bucket. Nothing has to be flushed; stale buckets expire.
  3. Within the bucket, the questions mean the same thing: cosine similarity
     of their embeddings is at least `min_similarity`, and they agree on the
     words embeddings blur -- numbers and identifiers ("10-32" vs "1/4-28"),
     negations ("not"), and limits/directions ("minimum" vs "maximum").

How a bucket is stored: one Redis list per bucket, newest entry first, capped
at `max_entries` and expiring after the TTL. A lookup reads the (short) list
and compares in Python; there is no vector index to keep in sync.

No question text is stored: an entry holds a hash of the normalised question,
a hash of its guard words, its embedding and the answer.

Redis down: a lookup is a miss and a store is skipped (logged). The cache is
an optimisation, never a dependency -- the same choice as rate limiting's
fail-open, behind the same kind of circuit breaker.
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.resilience import CircuitBreaker, CircuitOpenError

logger = structlog.get_logger(__name__)

# Any token with a digit: plain numbers, sizes and identifiers ("0.032",
# "1/4-28", "an960", "a320"). Two questions that differ here are about
# different things, however close their embeddings are.
_DIGIT_TOKEN = re.compile(r"[a-z]*\d[a-z0-9./-]*")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

_NEGATIONS = frozenset({"not", "no", "never", "without", "except", "cannot", "nor"})

# Words that flip a maintenance answer while barely moving the embedding, each
# mapped to one base form so "installed" and "installation" count as "install".
# Not a list of synonyms: a base form must simply appear in both questions or
# in neither.
_QUALIFIER_FORMS = {
    "min": ("min", "minimum", "minimal"),
    "max": ("max", "maximum", "maximal"),
    "upper": ("upper",),
    "lower": ("lower",),
    "above": ("above",),
    "below": ("below",),
    "before": ("before", "prior"),
    "after": ("after",),
    "more": ("more",),
    "less": ("less", "fewer"),
    "higher": ("higher",),
    "increase": ("increase", "increased", "increases", "increasing"),
    "decrease": ("decrease", "decreased", "decreases", "decreasing"),
    "inboard": ("inboard",),
    "outboard": ("outboard",),
    "left": ("left",),
    "right": ("right",),
    "forward": ("forward",),
    "aft": ("aft",),
    "front": ("front",),
    "rear": ("rear",),
    "open": ("open", "opened", "opening", "opens"),
    "close": ("close", "closed", "closing", "closes"),
    "install": ("install", "installed", "installing", "installs", "installation"),
    "remove": ("remove", "removed", "removing", "removes", "removal"),
    "tighten": ("tighten", "tightened", "tightening", "tightens"),
    "loosen": ("loosen", "loosened", "loosening", "loosens"),
    "first": ("first",),
    "last": ("last",),
    "hot": ("hot",),
    "cold": ("cold",),
    "wet": ("wet",),
    "dry": ("dry",),
}
_QUALIFIERS = {form: base for base, forms in _QUALIFIER_FORMS.items() for form in forms}


def normalize_question(question: str) -> str:
    """Lower-case, whitespace collapsed, trailing punctuation dropped."""
    return " ".join(question.lower().split()).rstrip(" ?.!")


def question_guard(question: str) -> tuple[str, ...]:
    """The words two questions must share before their embeddings are compared."""
    text = question.lower()
    guard = set(_DIGIT_TOKEN.findall(text))
    for word in _WORD.findall(text):
        if word.endswith("n't"):
            guard.add("not")
        elif word in _NEGATIONS:
            guard.add("not" if word == "cannot" else word)
        elif word in _QUALIFIERS:
            guard.add(_QUALIFIERS[word])
    return tuple(sorted(guard))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors (0.0 if either is all zeros)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class CacheHit:
    answer: str
    similarity: float
    prompt_version: str | None
    age_seconds: float


class AnswerCache:
    """Answers bucketed by exact context, matched by question meaning."""

    def __init__(self, redis: Redis, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = redis
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_seconds=self.settings.breaker_recovery_seconds,
        )

    def _redis_key(self, bucket: str) -> str:
        return f"{self.settings.answer_cache_key_prefix}:{bucket}"

    async def lookup(
        self, bucket: str, question: str, embedding: list[float]
    ) -> CacheHit | None:
        """The closest stored answer for this context and question, if close enough."""
        key = self._redis_key(bucket)
        limit = self.settings.answer_cache_max_entries

        async def call():
            return await self.redis.lrange(key, 0, limit - 1)

        try:
            raw_entries = await self.breaker.call(call)
        except (RedisError, OSError, CircuitOpenError) as exc:
            logger.warning("answer cache unavailable", error=str(exc), op="lookup")
            return None

        question_hash = _digest(normalize_question(question))
        guard_hash = _digest("\0".join(question_guard(question)))
        threshold = self.settings.answer_cache_min_similarity
        best: CacheHit | None = None
        for raw in raw_entries:
            try:
                entry = json.loads(raw)
                if entry["g"] != guard_hash:
                    continue
                if entry["q"] == question_hash:
                    similarity = 1.0
                else:
                    similarity = cosine_similarity(entry["e"], embedding)
                if similarity < threshold:
                    continue
                if best is None or similarity > best.similarity:
                    best = CacheHit(
                        answer=entry["a"],
                        similarity=similarity,
                        prompt_version=entry.get("pv"),
                        age_seconds=max(0.0, time.time() - entry.get("t", 0)),
                    )
            except (ValueError, KeyError, TypeError):
                continue  # a malformed entry is a miss, not an error
        return best

    async def store(
        self,
        bucket: str,
        question: str,
        embedding: list[float],
        answer: str,
        prompt_version: str | None,
    ) -> None:
        """Remember an answer. Best effort: the caller already has it."""
        entry = json.dumps(
            {
                "q": _digest(normalize_question(question)),
                "g": _digest("\0".join(question_guard(question))),
                "e": [round(x, 6) for x in embedding],
                "a": answer,
                "pv": prompt_version,
                "t": time.time(),
            }
        )
        key = self._redis_key(bucket)

        async def call():
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.lpush(key, entry)
                pipe.ltrim(key, 0, self.settings.answer_cache_max_entries - 1)
                pipe.expire(key, self.settings.answer_cache_ttl_seconds)
                await pipe.execute()

        try:
            await self.breaker.call(call)
        except (RedisError, OSError, CircuitOpenError) as exc:
            logger.warning("answer cache unavailable", error=str(exc), op="store")

    async def close(self) -> None:
        await self.redis.aclose()


def create_answer_cache(settings: Settings | None = None) -> AnswerCache:
    settings = settings or get_settings()
    redis = Redis.from_url(
        settings.redis_url,
        # A cache lookup sits on every answered query; a slow Redis must cost
        # at most half a second before the request goes on without it.
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
    )
    return AnswerCache(redis, settings)
