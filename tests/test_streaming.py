"""SSE streaming of /v1/query/stream and the Ollama streaming client.

No Ollama or Postgres needed: Ollama is replaced by httpx.MockTransport (for
the client) or by fakes patched into the router (for the endpoint). Endpoint
tests use the real app, auth and rate limiter, so they need Redis like the
rest of /v1.
"""

import asyncio
import json

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from starlette.requests import ClientDisconnect

from app.api.v1 import router as router_module
from app.core.config import get_settings
from app.core.resilience import CircuitOpenError
from app.core.sse import format_event
from app.llm.ollama import OllamaClient
from app.main import app
from app.services import generation
from tests.auth_helpers import bearer, make_token

# ------------------------------------------------------------ Ollama client


def _ndjson(*pieces: str, done: bool = True) -> bytes:
    lines = [json.dumps({"message": {"content": p}, "done": False}) for p in pieces]
    if done:
        lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return ("\n".join(lines) + "\n").encode()


def _client(handler, **settings_overrides) -> OllamaClient:
    settings = get_settings().model_copy(
        update={"ollama_max_retries": 2, **settings_overrides}
    )
    client = OllamaClient(settings)
    client._client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    return client


async def _collect(client: OllamaClient) -> list[str]:
    return [piece async for piece in client.chat_stream("system", "user")]


async def test_stream_yields_pieces_in_order():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, content=_ndjson("Check ", "the ", "filter [1]."))

    pieces = await _collect(_client(handler))
    # trailing whitespace is held back until more text follows, so it ends up
    # at the start of the next piece; the joined text is unchanged
    assert pieces == ["Check", " the", " filter [1]."]
    assert "".join(pieces) == "Check the filter [1]."


async def test_stream_equals_chat_after_whitespace_trim():
    """Joined stream == chat()'s .strip()'d text, however the model splits it."""
    raw = ["\n\n  ", "Check", " ", " the", "\n", "filter.", "  ", "\n"]

    def handler(request):
        return httpx.Response(200, content=_ndjson(*raw))

    pieces = await _collect(_client(handler))
    assert "".join(pieces) == "".join(raw).strip()
    assert all(p.strip() for p in pieces)  # never an empty/blank event


async def test_stream_keeps_inner_whitespace_between_pieces():
    def handler(request):
        return httpx.Response(200, content=_ndjson("a", "\n\n", "b"))

    assert "".join(await _collect(_client(handler))) == "a\n\nb"


async def test_stream_error_line_mid_stream_raises():
    content = _ndjson("Partial ", done=False) + b'{"error": "model crashed"}\n'

    def handler(request):
        return httpx.Response(200, content=content)

    client = _client(handler)
    received = []
    with pytest.raises(RuntimeError, match="model crashed"):
        async for piece in client.chat_stream("s", "u"):
            received.append(piece)
    assert received == ["Partial"]  # what arrived before the failure


async def test_stream_non_200_raises_with_body_and_is_not_retried():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(RuntimeError, match="model not found"):
        await _collect(_client(handler))
    assert calls == 1


async def test_stream_retries_a_connection_failure_before_any_token():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=_ndjson("ok"))

    assert await _collect(_client(handler)) == ["ok"]
    assert calls == 2


async def test_stream_does_not_retry_a_read_timeout():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("model went quiet", request=request)

    with pytest.raises(httpx.ReadTimeout):
        await _collect(_client(handler))
    assert calls == 1  # a retry would start a second generation


async def test_stream_fails_fast_when_circuit_is_open():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("down", request=request)

    client = _client(handler, breaker_failure_threshold=1, ollama_max_retries=1)
    with pytest.raises(httpx.ConnectError):
        await _collect(client)
    with pytest.raises(CircuitOpenError):
        await _collect(client)
    assert calls == 1  # second call never reached the network


class _TrackedStream(httpx.AsyncByteStream):
    """Response body that records whether it was closed."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


async def test_stream_closes_connection_when_consumer_stops_early():
    body = _TrackedStream(_ndjson("one ", "two ", "three").splitlines(keepends=True))

    def handler(request):
        return httpx.Response(200, stream=body)

    stream = _client(handler).chat_stream("s", "u")
    assert await anext(stream) == "one"
    assert not body.closed
    await stream.aclose()  # what a disconnecting caller does
    assert body.closed


# --------------------------------------------------------------- SSE format


def test_format_event_keeps_newlines_inside_one_data_line():
    wire = format_event("token", {"text": "line one\nline two — ünïcode"})
    assert wire.endswith("\n\n")
    head, data_line, _blank = wire.split("\n", 2)
    assert head == "event: token"
    assert data_line.startswith("data: ")
    assert "\n" not in data_line
    assert json.loads(data_line[len("data: ") :]) == {
        "text": "line one\nline two — ünïcode"
    }


# ------------------------------------------------------------ service layer


class _Chunk:
    def __init__(self, cid: int, content: str, source: str = "unit"):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": source}


class _Session:
    async def commit(self) -> None:
        pass


class _ExplodingOllama:
    def chat_stream(self, *args, **kwargs):
        raise AssertionError("the model must not be called")

    async def chat(self, *args, **kwargs):
        raise AssertionError("the model must not be called")


async def test_refused_query_can_never_be_streamed(monkeypatch):
    async def far_away(session, ollama, query, top_k=None, **kwargs):
        return [(_Chunk(1, "unrelated"), get_settings().max_distance + 0.2)]

    monkeypatch.setattr(generation, "retrieve", far_away)

    prepared = await generation.prepare_answer(
        _Session(), _ExplodingOllama(), "anything"
    )
    assert prepared.refused
    assert prepared.prompt is None
    with pytest.raises(ValueError):
        generation.stream_answer(_ExplodingOllama(), prepared)


async def test_prepared_answer_matches_what_answer_question_sends(monkeypatch):
    async def near(session, ollama, query, top_k=None, **kwargs):
        return [(_Chunk(7, "check the reservoir"), 0.1)]

    monkeypatch.setattr(generation, "retrieve", near)

    sent: list[tuple[str, str]] = []

    class Ollama:
        async def chat(self, system, user):
            sent.append((system, user))
            return "ok"

        def chat_stream(self, system, user):
            sent.append((system, user))

            async def gen():
                yield "ok"

            return gen()

    plain = await generation.answer_question(_Session(), Ollama(), "q?")
    prepared = await generation.prepare_answer(_Session(), Ollama(), "q?")
    assert [p async for p in generation.stream_answer(Ollama(), prepared)] == ["ok"]
    assert sent[0] == sent[1]  # same system prompt and same assembled prompt
    assert plain.prompt_version == prepared.prompt_version


# --------------------------------------------------------------- the endpoint


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        assert name.startswith("event: ") and data.startswith("data: "), block
        events.append((name[len("event: ") :], json.loads(data[len("data: ") :])))
    return events


class FakeStream:
    """Replaces generation.stream_answer inside the router."""

    def __init__(self, pieces=("Check ", "the filter ", "[1]."), fail_after=None):
        self.pieces = list(pieces)
        self.fail_after = fail_after
        self.calls = 0

    def __call__(self, ollama, prepared):
        self.calls += 1

        async def gen():
            for i, piece in enumerate(self.pieces):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("secret-host.internal exploded")
                yield piece

        return gen()


def _prepared(refused: bool = False) -> generation.PreparedAnswer:
    if refused:
        return generation.PreparedAnswer(
            question="q?", refused=True, chunks=[(_Chunk(2, "far away"), 0.9)]
        )
    return generation.PreparedAnswer(
        question="q?",
        refused=False,
        chunks=[(_Chunk(1, "reservoir text", "AMM-29"), 0.12)],
        system="sys",
        prompt="prompt",
        prompt_version="v9",
    )


@pytest.fixture
def stream_env(monkeypatch):
    fake = FakeStream()
    state = {"prepared": _prepared()}

    async def fake_prepare(session, ollama, question, top_k=None):
        return state["prepared"]

    monkeypatch.setattr(router_module, "prepare_answer", fake_prepare)
    monkeypatch.setattr(router_module, "stream_answer", fake)
    return fake, state


@pytest.mark.redis
async def test_stream_endpoint_sends_meta_tokens_done(client, stream_env):
    response = await client.post("/v1/query/stream", json={"question": "q?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "x-request-id" in response.headers

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names == ["meta", "token", "token", "token", "done"]

    meta = events[0][1]
    assert meta["refused"] is False
    assert meta["prompt_version"] == "v9"
    assert meta["chunks"][0]["source"] == "AMM-29"
    assert "answer" not in meta

    text = "".join(data["text"] for name, data in events if name == "token")
    assert text == "Check the filter [1]."
    assert events[-1][1] == {"refused": False, "answer_chars": len(text)}


@pytest.mark.redis
async def test_refusal_streams_meta_and_done_without_calling_the_model(
    client, stream_env
):
    fake, state = stream_env
    state["prepared"] = _prepared(refused=True)

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "done"]
    assert events[0][1]["refused"] is True
    assert "Refusing to answer" in events[0][1]["refusal_message"]
    assert events[1][1] == {"refused": True, "answer_chars": 0}
    assert fake.calls == 0  # structural: the model was never reached


@pytest.mark.redis
async def test_failure_mid_stream_is_reported_in_band_without_leaking(
    client, stream_env
):
    fake, _ = stream_env
    fake.fail_after = 1

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    assert response.status_code == 200  # headers were already sent
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "token", "error"]
    assert events[-1][1] == {"detail": "Answer generation failed."}
    assert "secret-host" not in response.text


@pytest.mark.redis
async def test_failure_before_streaming_is_a_normal_http_error(client, monkeypatch):
    async def broken_prepare(session, ollama, question, top_k=None):
        raise CircuitOpenError("circuit is open")

    monkeypatch.setattr(router_module, "prepare_answer", broken_prepare)
    with pytest.raises(CircuitOpenError):  # same handling as /v1/query today
        await client.post("/v1/query/stream", json={"question": "q?"})


@pytest.mark.redis
async def test_stream_requires_authentication(anon_client, stream_env):
    response = await anon_client.post("/v1/query/stream", json={"question": "q?"})
    assert response.status_code == 401
    assert stream_env[0].calls == 0


@pytest.mark.redis
async def test_stream_requires_query_scope(anon_client, stream_env):
    token = make_token(subject="ingest-only", scopes=("ingest",))
    response = await anon_client.post(
        "/v1/query/stream", json={"question": "q?"}, headers=bearer(token)
    )
    assert response.status_code == 403
    assert stream_env[0].calls == 0


@pytest.mark.redis
@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "q", "top_k": 0}])
async def test_stream_validates_the_body_before_streaming(client, stream_env, body):
    response = await client.post("/v1/query/stream", json=body)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.redis
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_client_disconnect_stops_generation_and_closes_the_stream(
    stream_env, monkeypatch, spec_version
):
    """Hang up after the first token: the model stream must be closed promptly.

    httpx's test transport buffers whole responses, so this drives the ASGI app
    directly and hangs up the way real servers do: ASGI 2.3 servers send
    `http.disconnect` through `receive`; 2.4+ servers make the next `send`
    raise OSError. Both must end generation.
    """
    state = {"started": False, "closed": False, "yielded": 0}

    def endless(ollama, prepared):
        async def gen():
            state["started"] = True
            try:
                while True:
                    state["yielded"] += 1
                    yield "tok "
                    await asyncio.sleep(0.01)
            finally:
                # runs under cancellation: must not need to await anything
                state["closed"] = True

        return gen()

    monkeypatch.setattr(router_module, "stream_answer", endless)

    body = json.dumps({"question": "q?"}).encode()
    disconnect = asyncio.Event()
    sent_body = False

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        if disconnect.is_set() and spec_version == "2.4":
            raise OSError("client went away")
        messages.append(message)
        if message["type"] == "http.response.body" and b"event: token" in message.get(
            "body", b""
        ):
            disconnect.set()

    headers = [
        (b"content-type", b"application/json"),
        (b"authorization", bearer(make_token())["Authorization"].encode()),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/query/stream",
        "raw_path": b"/v1/query/stream",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "state": {},
    }

    async with LifespanManager(app) as manager:
        with anyio.fail_after(5):
            try:
                await manager.app(scope, receive, send)
            except ClientDisconnect:
                pass  # Starlette's way of reporting the hang-up on ASGI 2.4+

    assert state["started"]
    assert state["closed"], "generator was left running after the client left"
    # it stopped soon after the hang-up instead of producing forever
    assert state["yielded"] < 50


# ---------------------------------------------------------- figure verification


def _with_passage(content: str) -> generation.PreparedAnswer:
    return generation.PreparedAnswer(
        question="q?",
        refused=False,
        chunks=[(_Chunk(1, content, "AC"), 0.12)],
        system="sys",
        prompt="prompt",
        prompt_version="v9",
    )


@pytest.mark.redis
async def test_stream_reports_a_figure_the_passage_does_not_contain(client, stream_env):
    fake, state = stream_env
    state["prepared"] = _with_passage("Torque the nut to 20 inch-pounds.")
    fake.pieces = ["Torque ", "to 25 ", "inch-pounds [1]."]

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    events = parse_sse(response.text)
    assert [name for name, _ in events] == [
        "meta",
        "token",
        "token",
        "token",
        "verification",
        "done",
    ]
    verification = dict(events)["verification"]
    assert verification["ok"] is False
    assert verification["unsupported"] == ["25 inch-pounds"]
    assert verification["blocked"] is False  # flag mode: nothing to discard


@pytest.mark.redis
async def test_stream_confirms_a_figure_the_passage_contains(client, stream_env):
    fake, state = stream_env
    state["prepared"] = _with_passage("Torque the nut to 20 inch-pounds.")
    fake.pieces = ["Torque to 20 ", "inch-pounds [1]."]

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    verification = dict(parse_sse(response.text))["verification"]
    assert verification["ok"] is True
    assert verification["checked"] == 1


@pytest.mark.redis
async def test_stream_in_block_mode_tells_the_client_to_discard(
    client, stream_env, monkeypatch
):
    fake, state = stream_env
    state["prepared"] = _with_passage("Torque the nut to 20 inch-pounds.")
    fake.pieces = ["Torque to 25 inch-pounds [1]."]
    blocking = get_settings().model_copy(update={"verify_mode": "block"})
    monkeypatch.setattr(generation, "get_settings", lambda: blocking)
    monkeypatch.setattr(router_module, "get_settings", lambda: blocking)

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    assert dict(parse_sse(response.text))["verification"]["blocked"] is True


@pytest.mark.redis
async def test_stream_with_verification_off_sends_no_verification(
    client, stream_env, monkeypatch
):
    fake, state = stream_env
    state["prepared"] = _with_passage("Torque the nut to 20 inch-pounds.")
    fake.pieces = ["Torque to 25 inch-pounds [1]."]
    off = get_settings().model_copy(update={"verify_mode": "off"})
    monkeypatch.setattr(generation, "get_settings", lambda: off)
    monkeypatch.setattr(router_module, "get_settings", lambda: off)

    response = await client.post("/v1/query/stream", json={"question": "q?"})

    assert "verification" not in [name for name, _ in parse_sse(response.text)]
