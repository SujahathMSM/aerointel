import asyncio

import pytest

from app.core.resilience import CircuitBreaker, CircuitOpenError, retry


async def test_retry_returns_first_try():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry(fn, attempts=3, base_delay=0.01) == "ok"
    assert calls == 1


async def test_retry_recovers_from_transient_failures():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient")
        return "ok"

    assert await retry(fn, attempts=3, base_delay=0.01) == "ok"
    assert calls == 3


async def test_retry_raises_after_last_attempt():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        await retry(fn, attempts=3, base_delay=0.01)
    assert calls == 3


async def test_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=60)

    async def failing():
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(failing)

    assert breaker.state == "open"

    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)  # fails fast, fn never runs


async def test_breaker_recovers_after_probe_success():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)

    async def failing():
        raise RuntimeError("down")

    async def working():
        return "ok"

    with pytest.raises(RuntimeError):
        await breaker.call(failing)
    assert breaker.state == "open"

    await asyncio.sleep(0.06)
    assert breaker.state == "half_open"

    assert await breaker.call(working) == "ok"
    assert breaker.state == "closed"


async def test_breaker_reopens_after_failed_probe():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)

    async def failing():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await breaker.call(failing)

    await asyncio.sleep(0.06)
    with pytest.raises(RuntimeError):
        await breaker.call(failing)  # probe failed

    assert breaker.state == "open"
