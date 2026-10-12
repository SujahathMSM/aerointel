"""RequestIdMiddleware is plain ASGI (not BaseHTTPMiddleware) so it can sit in
front of streaming responses. These tests drive it directly, without the app,
so they need no database, Redis or Ollama."""

import asyncio

import httpx
import structlog
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route

from app.core.logging import RequestIdMiddleware


def _client(routes) -> httpx.AsyncClient:
    app = RequestIdMiddleware(Starlette(routes=routes))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _ok(request):
    return PlainTextResponse("ok")


async def test_generates_an_id_when_none_supplied():
    async with _client([Route("/", _ok)]) as client:
        response = await client.get("/")
    assert len(response.headers["x-request-id"]) == 12


async def test_echoes_a_safe_client_supplied_id():
    async with _client([Route("/", _ok)]) as client:
        response = await client.get("/", headers={"x-request-id": "trace-abc.123"})
    assert response.headers["x-request-id"] == "trace-abc.123"


async def test_replaces_an_unsafe_client_supplied_id():
    # Spaces/oversize values must not be copied into headers and log lines.
    async with _client([Route("/", _ok)]) as client:
        for bad in ("has space", "x" * 200, "semi;colon"):
            response = await client.get("/", headers={"x-request-id": bad})
            assert response.headers["x-request-id"] != bad
            assert len(response.headers["x-request-id"]) == 12


async def test_streaming_body_arrives_chunk_by_chunk():
    # Regression guard for SSE (arriving later in Phase 2): the middleware must
    # pass body chunks through as they are produced, not hold them until the
    # response ends. Driven directly because httpx's in-process transport
    # buffers whole bodies and so cannot observe this. The producer pauses
    # after chunk one until the client side of the ASGI conversation has
    # actually received it.
    first_chunk_sent = asyncio.Event()
    producer_saw_chunk_arrive = False

    async def body():
        nonlocal producer_saw_chunk_arrive
        yield b"data: one\n\n"
        try:
            await asyncio.wait_for(first_chunk_sent.wait(), timeout=2)
            producer_saw_chunk_arrive = True
        except TimeoutError:
            pass
        yield b"data: two\n\n"

    async def stream(request):
        return StreamingResponse(body(), media_type="text/event-stream")

    app = RequestIdMiddleware(Starlette(routes=[Route("/s", stream)]))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/s",
        "raw_path": b"/s",
        "query_string": b"",
        "headers": [],
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
    }
    messages: list[dict] = []

    async def receive():
        await asyncio.sleep(3600)  # never disconnects within the test
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()

    await app(scope, receive, send)

    start = messages[0]
    assert start["type"] == "http.response.start"
    assert dict(start["headers"])[b"x-request-id"]
    assert producer_saw_chunk_arrive, "first chunk was held back until the end"
    bodies = [m["body"] for m in messages if m["type"] == "http.response.body"]
    assert b"".join(bodies) == b"data: one\n\ndata: two\n\n"


async def test_request_id_is_bound_to_logs_and_cleared_afterwards():
    seen = {}

    async def handler(request):
        seen["during"] = structlog.contextvars.get_contextvars().get("request_id")
        seen["state"] = request.scope["state"]["request_id"]
        return PlainTextResponse("ok")

    async with _client([Route("/", handler)]) as client:
        response = await client.get("/")

    assert seen["during"] == response.headers["x-request-id"] == seen["state"]
    assert "request_id" not in structlog.contextvars.get_contextvars()
