import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Circuit breaker is open: downstream is known-bad, failing fast."""


async def retry(
    fn: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call fn, retrying with exponential backoff (0.5s, 1s, 2s, ...)."""
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "call failed, retrying",
                attempt=attempt + 1,
                delay_seconds=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # for the type checker


class CircuitBreaker:
    """Stop calling a failing service for a cool-down window.

    closed -> normal. N consecutive failures -> open (fail fast).
    After recovery_seconds -> half-open: one probe call decides
    whether to close or re-open.
    """

    def __init__(
        self, failure_threshold: int = 5, recovery_seconds: float = 30.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self.recovery_seconds:
            return "half_open"
        return "open"

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        state = self.state
        if state == "open":
            raise CircuitOpenError("circuit is open, failing fast")

        try:
            result = await fn()
        except Exception:
            if state == "half_open":
                # probe failed: re-open for another window
                self._opened_at = time.monotonic()
                logger.warning("circuit re-opened after failed probe")
            else:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "circuit opened", consecutive_failures=self._failures
                    )
            raise

        self._failures = 0
        self._opened_at = None
        return result
