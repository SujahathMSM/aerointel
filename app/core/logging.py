import logging
import re
import uuid
from contextlib import contextmanager

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def configure_logging() -> None:
    """JSON logs with named fields, instead of prose lines."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


# A client may supply its own ID so one trace can span several services. Only
# accept something that is safe to put in a header and a log line; anything
# else (too long, spaces, control characters) is replaced with a fresh ID.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Give every request an ID; all logs for it carry the same ID.

    Written as plain ASGI, not on BaseHTTPMiddleware. BaseHTTPMiddleware runs
    the downstream app in a separate task and re-wraps the response body, which
    is a known source of trouble for long-lived streaming responses (SSE,
    arriving later in Phase 2): client disconnects and cancellation do not
    reliably reach the endpoint, and contextvars bound downstream do not
    propagate back. Plain ASGI adds nothing between the server and the app: it
    only adds a header to the `http.response.start` message and passes every
    other message through untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan, websocket: nothing to stamp
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-request-id")
        if supplied and _SAFE_REQUEST_ID.match(supplied):
            request_id = supplied
        else:
            request_id = uuid.uuid4().hex[:12]

        # Handlers can read it from here (e.g. to put it in an error body).
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            structlog.contextvars.clear_contextvars()


@contextmanager
def bind_log_context(**fields):
    """Bind extra fields to logs inside a block (e.g. chunk_id=...)."""
    structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*fields.keys())
