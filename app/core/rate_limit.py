"""Token-bucket rate limiting backed by Redis (Phase 2, step 1).

Why a token bucket: every client gets a bucket holding up to `capacity`
tokens, refilled at `refill_per_second`. Each request spends one token; an
empty bucket means 429. Short bursts (up to `capacity`) pass untouched,
while the long-run rate is capped at `refill_per_second`.

Why Redis and not a dict in the process: the bucket has to be shared. A
per-process dict gives every uvicorn worker (and every API replica, later)
its own separate budget, so N workers silently allow N times the limit.
Redis is the one place all of them agree on.

Why a Lua script: "read tokens -> refill -> spend -> write back" must be
atomic. Done as separate GET/SET calls, two concurrent requests can both read
"1 token left" and both get through. Redis runs a Lua script as a single
uninterruptible step, so that race cannot happen. The script also takes the
time from Redis itself (TIME), so API replicas with drifting clocks still
agree on how much a bucket has refilled.
"""

import math
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.resilience import CircuitBreaker, CircuitOpenError

logger = structlog.get_logger(__name__)

# KEYS[1] = bucket key
# ARGV[1] = capacity, ARGV[2] = refill per second, ARGV[3] = cost
# Returns {allowed (0/1), tokens_left, retry_after_seconds}. The last two
# are returned as strings because Redis truncates Lua numbers to integers.
TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / refill
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
-- An idle bucket is full again after capacity/refill seconds; past that,
-- keeping the key is pointless, so let Redis drop it.
redis.call('EXPIRE', key, math.ceil(capacity / refill) + 1)

return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    # True when Redis was unreachable and the request was let through
    # anyway (fail-open). Surfaced so tests and logs can tell a real
    # "allowed" from a degraded one.
    degraded: bool = False


class RateLimiter:
    """Per-client token bucket stored in Redis.

    Redis calls sit behind a short socket timeout and a circuit breaker, so
    a dead Redis costs at most a few fast failures and then nothing: the
    breaker opens and every request skips Redis until the probe window.
    What happens to the request then is `fail_open`:
      True  -> allow it (the limiter is protection, not a dependency; a
               Redis outage should not take the whole API down with it)
      False -> reject it with 503 (strict mode: no limit, no service)
    """

    def __init__(self, redis: Redis, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redis = redis
        self.capacity = self.settings.rate_limit_capacity
        self.refill_per_second = self.settings.rate_limit_refill_per_second
        self.key_prefix = self.settings.rate_limit_key_prefix
        self.fail_open = self.settings.rate_limit_fail_open
        self._script = redis.register_script(TOKEN_BUCKET_LUA)
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_seconds=self.settings.breaker_recovery_seconds,
        )

    async def hit(self, identity: str, cost: int = 1) -> RateLimitResult:
        """Spend `cost` tokens from `identity`'s bucket."""
        key = f"{self.key_prefix}:{identity}"

        async def call():
            return await self._script(
                keys=[key], args=[self.capacity, self.refill_per_second, cost]
            )

        try:
            allowed, tokens_left, retry_after = await self.breaker.call(call)
        except (RedisError, OSError, CircuitOpenError) as exc:
            logger.warning(
                "rate limiter unavailable",
                error=str(exc),
                fail_open=self.fail_open,
                breaker_state=self.breaker.state,
            )
            return RateLimitResult(
                allowed=self.fail_open,
                limit=self.capacity,
                remaining=0,
                retry_after_seconds=0,
                degraded=True,
            )

        return RateLimitResult(
            allowed=bool(int(allowed)),
            limit=self.capacity,
            remaining=int(float(tokens_left)),
            # Retry-After is whole seconds; round up so a client that waits
            # exactly that long is guaranteed a token.
            retry_after_seconds=math.ceil(float(retry_after)),
        )

    async def close(self) -> None:
        await self.redis.aclose()


def create_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    settings = settings or get_settings()
    redis = Redis.from_url(
        settings.redis_url,
        # Rate limiting sits in front of every /v1 request, so a slow Redis
        # must not become a slow API. Half a second, then fail fast.
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
    )
    return RateLimiter(redis, settings)
