"""The semantic answer cache: what may be reused, and what never is.

The properties that matter, in order:
  - a refused question never reaches the cache (refusal stays structural);
  - a hit needs the identical context, so citations always point at the
    passages this question retrieved;
  - questions that differ in a number, a negation or a limit never share;
  - the figure check runs on every answer, cached or not, and a flagged
    answer is never handed to the next caller;
  - Redis down means "no cache", not an error.

Unit tests use a fake cache; the `redis`-marked ones use the real Redis
(docker compose locally, the service container in CI), like the other
Phase 2 stores.
"""

import json
import uuid

import pytest
from redis.asyncio import Redis

from app.core.answer_cache import (
    AnswerCache,
    CacheHit,
    cosine_similarity,
    create_answer_cache,
    normalize_question,
    question_guard,
)
from app.core.config import Settings, get_settings
from app.main import app
from app.services import generation
from app.services.verification import Verification
from tests.test_streaming import parse_sse

# ------------------------------------------------------------------ helpers


class _Chunk:
    def __init__(self, cid: int, content: str, source: str = "AC 43.13-1B"):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": source}


TORQUE = _Chunk(1, "Torque a 10-32 bolt to 20-25 inch-pounds.")
SAFETY = _Chunk(2, "Safety wire the bolt head after torquing.")


class _Session:
    async def commit(self) -> None:
        pass


def _vector(*values: float) -> list[float]:
    return [*values, 0.0, 0.0]


# Embeddings the fake model hands out: the two paraphrases are nearly
# parallel (cosine ~0.995), the unrelated question is far from both.
EMBEDDINGS = {
    "What torque for a 10-32 bolt?": _vector(1.0, 0.0),
    "How tight should a 10-32 bolt be torqued?": _vector(1.0, 0.1),
    "Which safety wire size is used?": _vector(0.0, 1.0),
}


class _Llm:
    """Embeds from EMBEDDINGS; records every answer it is asked to write."""

    model_name = "fake-model"

    def __init__(self, answer: str = "Torque it to 20-25 inch-pounds [1]."):
        self.answer = answer
        self.chats = 0
        self.streams = 0
        self.embeds = 0

    async def embed(self, texts):
        self.embeds += 1
        return [EMBEDDINGS.get(t, _vector(0.5, 0.5)) for t in texts]

    async def chat(self, system, user):
        self.chats += 1
        return self.answer

    def chat_stream(self, system, user):
        self.streams += 1

        async def gen():
            for piece in ("Torque it ", "to 20-25 ", "inch-pounds [1]."):
                yield piece

        return gen()

    async def close(self):
        pass


class _FakeCache:
    """In-memory stand-in with the same lookup rules as the real one."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.entries: dict[str, list[tuple[str, list[float], str, str | None]]] = {}
        self.lookups = 0
        self.stores = 0

    async def lookup(self, bucket, question, embedding):
        self.lookups += 1
        best = None
        for q, emb, answer, pv in self.entries.get(bucket, []):
            if question_guard(q) != question_guard(question):
                continue
            sim = cosine_similarity(emb, embedding)
            if sim >= self.settings.answer_cache_min_similarity and (
                best is None or sim > best.similarity
            ):
                best = CacheHit(answer, sim, pv, 0.0)
        return best

    async def store(self, bucket, question, embedding, answer, prompt_version):
        self.stores += 1
        self.entries.setdefault(bucket, []).insert(
            0, (question, embedding, answer, prompt_version)
        )


@pytest.fixture
def cache_on(monkeypatch):
    """Answer cache switched on, retrieval faked to return `state['chunks']`."""
    values = get_settings().model_copy(update={"answer_cache_enabled": True})
    monkeypatch.setattr(generation, "get_settings", lambda: values)
    state: dict = {"chunks": [(TORQUE, 0.12), (SAFETY, 0.30)], "embeddings": []}

    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        state["embeddings"].append(kwargs.get("query_embedding"))
        return state["chunks"]

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    return state


async def _ask(llm, cache, question):
    return await generation.answer_question(_Session(), llm, question, cache=cache)


# ------------------------------------------------------------ guard words


def test_numbers_and_identifiers_are_guard_words():
    assert question_guard("Torque for a 10-32 bolt?") != question_guard(
        "Torque for a 1/4-28 bolt?"
    )
    assert question_guard("AN960 washer") != question_guard("AN970 washer")
    assert "0.032" in question_guard("Is 0.032 in. sheet enough?")


def test_negations_and_limits_are_guard_words():
    assert question_guard("Should I prime it?") != question_guard(
        "Should I not prime it?"
    )
    assert question_guard("Can't I prime it?") == question_guard("Cannot I prime it?")
    assert question_guard("minimum edge distance") != question_guard(
        "maximum edge distance"
    )
    # inflected forms count as their base word
    assert question_guard("when it is installed") == question_guard("at installation")
    assert question_guard("when it is installed") != question_guard(
        "when it is removed"
    )
    assert question_guard("min edge distance") == question_guard(
        "minimum edge distance"
    )


def test_ordinary_wording_is_not_a_guard_word():
    assert question_guard("What torque for a 10-32 bolt?") == question_guard(
        "How tight should a 10-32 bolt be torqued"
    )


def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_question("  What   torque? ") == normalize_question("what torque")


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0  # different models


# --------------------------------------------------------------- the bucket


def _prepared(chunks, system="sys v1"):
    return generation.PreparedAnswer(
        question="q?", refused=False, chunks=chunks, system=system, prompt="p"
    )


def test_bucket_ignores_the_question_but_not_the_context():
    llm = _Llm()
    base = generation.cache_bucket(_prepared([(TORQUE, 0.1), (SAFETY, 0.3)]), llm)

    other_question = _prepared([(TORQUE, 0.1), (SAFETY, 0.3)])
    other_question.question = "something else entirely?"
    assert generation.cache_bucket(other_question, llm) == base
    # distances are not part of what the model reads
    assert generation.cache_bucket(_prepared([(TORQUE, 0.2), (SAFETY, 0.4)]), llm) == (
        base
    )

    # order changes what [1] and [2] mean
    swapped = _prepared([(SAFETY, 0.3), (TORQUE, 0.1)])
    assert generation.cache_bucket(swapped, llm) != base
    fewer = _prepared([(TORQUE, 0.1)])
    assert generation.cache_bucket(fewer, llm) != base
    edited = _Chunk(1, "Torque a 10-32 bolt to 23-25 inch-pounds.")
    assert generation.cache_bucket(_prepared([(edited, 0.1), (SAFETY, 0.3)]), llm) != (
        base
    )
    new_prompt = _prepared([(TORQUE, 0.1), (SAFETY, 0.3)], system="sys v2")
    assert generation.cache_bucket(new_prompt, llm) != base


def test_bucket_changes_with_model_and_settings(monkeypatch):
    prepared = _prepared([(TORQUE, 0.1)])
    base = generation.cache_bucket(prepared, _Llm())

    other_model = _Llm()
    other_model.model_name = "another-model"
    assert generation.cache_bucket(prepared, other_model) != base

    for change in (
        {"embedding_model": "other-embedder"},
        {"temperature": 0.9},
        {"llm_num_ctx": 8192},
    ):
        values = get_settings().model_copy(update=change)
        monkeypatch.setattr(generation, "get_settings", lambda v=values: v)
        assert generation.cache_bucket(prepared, _Llm()) != base, change


# ------------------------------------------------------ the answer service


async def test_second_equivalent_question_reuses_the_answer(cache_on):
    llm, cache = _Llm(), _FakeCache()

    first = await _ask(llm, cache, "What torque for a 10-32 bolt?")
    second = await _ask(llm, cache, "How tight should a 10-32 bolt be torqued?")

    assert llm.chats == 1  # the second answer came from the cache
    assert first.cached is False and second.cached is True
    assert second.answer == first.answer
    assert second.chunks == first.chunks
    assert second.verification is not None and second.verification.ok


async def test_embedding_is_computed_once_and_reused_by_retrieval(cache_on):
    llm = _Llm()
    await _ask(llm, _FakeCache(), "What torque for a 10-32 bolt?")
    assert llm.embeds == 1
    assert cache_on["embeddings"] == [EMBEDDINGS["What torque for a 10-32 bolt?"]]


async def test_unrelated_question_over_the_same_passages_misses(cache_on):
    llm, cache = _Llm(), _FakeCache()
    await _ask(llm, cache, "What torque for a 10-32 bolt?")
    result = await _ask(llm, cache, "Which safety wire size is used?")
    assert llm.chats == 2 and result.cached is False


async def test_different_context_misses_even_for_the_same_question(cache_on):
    llm, cache = _Llm(), _FakeCache()
    await _ask(llm, cache, "What torque for a 10-32 bolt?")
    cache_on["chunks"] = [(TORQUE, 0.12)]  # e.g. a document was replaced
    result = await _ask(llm, cache, "What torque for a 10-32 bolt?")
    assert llm.chats == 2 and result.cached is False


async def test_a_refused_question_never_touches_the_cache(cache_on):
    llm, cache = _Llm(), _FakeCache()
    await _ask(llm, cache, "What torque for a 10-32 bolt?")
    cache_on["chunks"] = [(TORQUE, get_settings().max_distance + 0.1)]

    result = await _ask(llm, cache, "What torque for a 10-32 bolt?")

    assert result.refused is True and result.answer is None
    assert result.cached is False
    assert cache.lookups == 1  # only the first, answered question looked


async def test_a_flagged_answer_is_returned_but_not_stored(cache_on):
    llm, cache = _Llm(answer="Torque it to 40 inch-pounds [1]."), _FakeCache()
    result = await _ask(llm, cache, "What torque for a 10-32 bolt?")
    assert result.verification is not None and not result.verification.ok
    assert result.answer is not None  # flag mode: the caller still gets it
    assert cache.stores == 0


async def test_verification_runs_again_on_a_hit(cache_on, monkeypatch):
    llm, cache = _Llm(), _FakeCache()
    await _ask(llm, cache, "What torque for a 10-32 bolt?")
    checked: list[str] = []

    def spy(prepared, answer):
        checked.append(answer)
        return Verification(checked=1, unsupported=["20-25"])

    monkeypatch.setattr(generation, "verify_prepared", spy)
    values = generation.get_settings().model_copy(update={"verify_mode": "block"})
    monkeypatch.setattr(generation, "get_settings", lambda: values)

    result = await _ask(llm, cache, "How tight should a 10-32 bolt be torqued?")
    assert checked == ["Torque it to 20-25 inch-pounds [1]."]
    assert result.refused is True  # block mode withholds a cached answer too
    assert result.refusal_reason == generation.REASON_UNVERIFIED


async def test_without_a_cache_nothing_changes(cache_on):
    llm = _Llm()
    await _ask(llm, None, "What torque for a 10-32 bolt?")
    result = await _ask(llm, None, "What torque for a 10-32 bolt?")
    assert llm.chats == 2 and result.cached is False


# ----------------------------------------------------------- real Redis


def _settings(**overrides) -> Settings:
    base = get_settings().model_dump()
    base.update(answer_cache_key_prefix=f"test-cache-unit:{uuid.uuid4().hex}")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
async def make_cache():
    created: list[AnswerCache] = []

    def _make(**overrides) -> AnswerCache:
        cache = create_answer_cache(_settings(**overrides))
        created.append(cache)
        return cache

    yield _make
    for cache in created:
        await cache.close()


Q1 = "What torque for a 10-32 bolt?"
Q2 = "How tight should a 10-32 bolt be torqued?"


@pytest.mark.redis
async def test_store_then_lookup(make_cache):
    cache = make_cache()
    await cache.store("b", Q1, EMBEDDINGS[Q1], "20-25 in-lb [1].", "v1")

    exact = await cache.lookup("b", "what TORQUE for a 10-32 bolt", [9.0, 9.0, 9, 9])
    assert exact is not None and exact.similarity == 1.0  # by text, not vector
    assert exact.answer == "20-25 in-lb [1]." and exact.prompt_version == "v1"

    close = await cache.lookup("b", Q2, EMBEDDINGS[Q2])
    assert close is not None and close.similarity == pytest.approx(0.995, abs=1e-3)

    assert await cache.lookup("other-bucket", Q1, EMBEDDINGS[Q1]) is None
    far = "Which safety wire size is used?"
    assert await cache.lookup("b", far, EMBEDDINGS[far]) is None


@pytest.mark.redis
async def test_guard_words_block_even_identical_vectors(make_cache):
    cache = make_cache()
    await cache.store("b", Q1, EMBEDDINGS[Q1], "a", None)
    other_size = "What torque for a 1/4-28 bolt?"
    assert await cache.lookup("b", other_size, EMBEDDINGS[Q1]) is None


@pytest.mark.redis
async def test_threshold_is_configurable(make_cache):
    strict = make_cache(answer_cache_min_similarity=0.999)
    await strict.store("b", Q1, EMBEDDINGS[Q1], "a", None)
    assert await strict.lookup("b", Q2, EMBEDDINGS[Q2]) is None


@pytest.mark.redis
async def test_buckets_are_capped_and_expire(make_cache):
    cache = make_cache(answer_cache_max_entries=3, answer_cache_ttl_seconds=60)
    for i in range(5):
        await cache.store("b", f"question {i}?", _vector(1.0, i), f"a{i}", None)

    key = cache._redis_key("b")
    entries = [json.loads(e) for e in await cache.redis.lrange(key, 0, -1)]
    assert [e["a"] for e in entries] == ["a4", "a3", "a2"]  # newest kept
    assert 0 < await cache.redis.ttl(key) <= 60
    # no question text is stored, only hashes and the vector
    assert all("question" not in json.dumps(e) for e in entries)


@pytest.mark.redis
async def test_malformed_entries_are_skipped(make_cache):
    cache = make_cache()
    await cache.redis.lpush(cache._redis_key("b"), "not json", json.dumps({"a": 1}))
    await cache.store("b", Q1, EMBEDDINGS[Q1], "ok", None)
    hit = await cache.lookup("b", Q1, EMBEDDINGS[Q1])
    assert hit is not None and hit.answer == "ok"


@pytest.mark.redis
async def test_redis_down_is_a_miss_not_an_error():
    cache = AnswerCache(
        Redis.from_url(
            "redis://127.0.0.1:1/0", socket_timeout=0.2, socket_connect_timeout=0.2
        ),
        _settings(),
    )
    try:
        await cache.store("b", Q1, EMBEDDINGS[Q1], "a", None)  # must not raise
        assert await cache.lookup("b", Q1, EMBEDDINGS[Q1]) is None
        for _ in range(3):
            await cache.lookup("b", Q1, EMBEDDINGS[Q1])
        assert cache.breaker.state == "open"  # stops trying after a few failures
    finally:
        await cache.close()


# ------------------------------------------------------------ the endpoints


@pytest.fixture
async def api_with_cache(client, cache_on, make_cache):
    # Saved and restored by hand: monkeypatch would delete these on undo,
    # because Starlette's State keeps attributes outside the instance __dict__.
    llm = _Llm()
    original = (app.state.ollama, app.state.answer_cache)
    app.state.ollama, app.state.answer_cache = llm, make_cache()
    yield client, llm
    app.state.ollama, app.state.answer_cache = original


@pytest.mark.redis
async def test_query_endpoint_marks_a_cached_answer(api_with_cache):
    client, llm = api_with_cache

    first = (await client.post("/v1/query", json={"question": Q1})).json()
    second = (await client.post("/v1/query", json={"question": Q2})).json()

    assert llm.chats == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["model"] == "fake-model"
    assert second["verification"]["ok"] is True


@pytest.mark.redis
async def test_stream_reuses_and_feeds_the_cache(api_with_cache):
    client, llm = api_with_cache

    first = parse_sse(
        (await client.post("/v1/query/stream", json={"question": Q1})).text
    )
    assert first[0][1]["cached"] is False
    assert llm.streams == 1

    # what the stream stored, the plain endpoint can reuse, and the other way round
    plain = (await client.post("/v1/query", json={"question": Q2})).json()
    assert plain["cached"] is True and llm.chats == 0

    second = parse_sse(
        (await client.post("/v1/query/stream", json={"question": Q2})).text
    )
    names = [name for name, _ in second]
    assert names == ["meta", "token", "verification", "done"]
    assert second[0][1]["cached"] is True
    assert second[1][1]["text"] == "Torque it to 20-25 inch-pounds [1]."
    assert second[-1][1] == {"refused": False, "answer_chars": len(plain["answer"])}
    assert llm.streams == 1  # the model was not asked again


@pytest.mark.redis
async def test_refused_stream_is_not_cached(api_with_cache, monkeypatch):
    client, llm = api_with_cache
    cache = app.state.answer_cache
    far = [(TORQUE, get_settings().max_distance + 0.2)]

    async def far_away(session, ollama, query, top_k=None, **kwargs):
        return far

    monkeypatch.setattr(generation, "retrieve", far_away)

    events = parse_sse(
        (await client.post("/v1/query/stream", json={"question": Q1})).text
    )
    assert [name for name, _ in events] == ["meta", "done"]
    assert events[0][1]["cached"] is False
    assert llm.streams == 0
    assert await cache.redis.keys(f"{cache.settings.answer_cache_key_prefix}:*") == []
