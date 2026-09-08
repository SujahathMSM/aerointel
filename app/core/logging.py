import logging
import uuid
from contextlib import contextmanager

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


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


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Give every request an ID; all logs for it carry the same ID."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
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
