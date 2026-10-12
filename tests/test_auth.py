"""Phase 2, step 2: JWT authentication and scopes.

No database, Ollama or Redis needed for most of these: auth runs before any
endpoint logic, so a request the auth layer rejects never reaches them, and
an accepted request with an invalid body (422) proves auth let it through.
"""

import time

import jwt
import pytest

from app.core import auth
from app.core.auth import (
    AuthConfigError,
    TokenError,
    create_access_token,
    decode_access_token,
    parse_clients,
    validate_auth_settings,
    verify_client,
)
from app.core.config import Settings, get_settings
from tests.auth_helpers import TEST_CLIENTS, TEST_JWT_SECRET, bearer, make_token


def _settings(**overrides) -> Settings:
    base = get_settings().model_dump()
    base.update(overrides)
    return Settings(**base)


# ------------------------------------------------------------ token round trip


def test_token_round_trip_carries_subject_and_scopes():
    token, ttl = create_access_token(get_settings(), "alice", ["query"])
    principal = decode_access_token(get_settings(), token)

    assert principal.subject == "alice"
    assert principal.scopes == frozenset({"query"})
    assert ttl == get_settings().jwt_ttl_seconds


def _forge(settings: Settings, **claim_overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "sub": "mallory",
        "scope": "query ingest",
        "iat": now,
        "exp": now + 600,
        **claim_overrides,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def test_expired_token_is_rejected():
    token = _forge(get_settings(), exp=int(time.time()) - 3600)
    with pytest.raises(TokenError, match="expired"):
        decode_access_token(get_settings(), token)


@pytest.mark.parametrize("claim", ["exp", "iat", "sub", "iss", "aud"])
def test_every_required_claim_is_actually_required(claim):
    # A token with no `exp` would be valid forever, so absence must fail.
    token = _forge(get_settings(), **{claim: None})
    with pytest.raises(TokenError):
        decode_access_token(get_settings(), token)


def test_wrong_issuer_and_audience_are_rejected():
    with pytest.raises(TokenError):
        decode_access_token(get_settings(), _forge(get_settings(), iss="someone-else"))
    with pytest.raises(TokenError):
        decode_access_token(get_settings(), _forge(get_settings(), aud="other-api"))


def test_token_signed_with_another_secret_is_rejected():
    other = _settings(jwt_secret="a-completely-different-secret-" + "z" * 20)
    with pytest.raises(TokenError):
        decode_access_token(get_settings(), _forge(other))


def test_alg_none_token_is_rejected():
    # The classic bypass: an unsigned token claiming `alg: none`.
    now = int(time.time())
    unsigned = jwt.encode(
        {
            "iss": get_settings().jwt_issuer,
            "aud": get_settings().jwt_audience,
            "sub": "mallory",
            "scope": "ingest",
            "iat": now,
            "exp": now + 600,
        },
        key=None,
        algorithm="none",
    )
    with pytest.raises(TokenError):
        decode_access_token(get_settings(), unsigned)


@pytest.mark.filterwarnings("ignore:The HMAC key:Warning")
def test_token_with_a_different_algorithm_is_rejected():
    # Verification is pinned to the configured algorithm, not to whatever the
    # token's own header asks for.
    settings = get_settings()
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "sub": "mallory",
            "iat": now,
            "exp": now + 600,
        },
        settings.jwt_secret,  # the RIGHT secret: only the algorithm differs
        algorithm="HS512",
    )
    with pytest.raises(TokenError):
        decode_access_token(settings, token)


def test_garbage_is_rejected():
    for junk in ("", "not-a-jwt", "a.b.c", "Bearer abc"):
        with pytest.raises(TokenError):
            decode_access_token(get_settings(), junk)


# ------------------------------------------------------------ client credentials


def test_verify_client_accepts_right_secret_only():
    settings = get_settings()
    good = TEST_CLIENTS["test-reader"]["secret"]
    assert verify_client(settings, "test-reader", good) is not None
    assert verify_client(settings, "test-reader", good + "x") is None
    assert verify_client(settings, "nobody", good) is None


def test_secret_comparison_is_constant_time(monkeypatch):
    calls = []
    real = auth.hmac.compare_digest
    monkeypatch.setattr(
        auth.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b)
    )
    verify_client(get_settings(), "no-such-client", "whatever-secret-value")
    assert calls, "unknown ids must still cost a comparison"


# --------------------------------------------------------------- startup checks


def test_startup_refuses_a_missing_or_short_secret():
    for secret in (None, "", "too-short"):
        with pytest.raises(AuthConfigError, match="JWT_SECRET"):
            validate_auth_settings(_settings(jwt_secret=secret))


def test_startup_allows_no_secret_when_auth_is_off():
    validate_auth_settings(_settings(auth_enabled=False, jwt_secret=None))


def test_startup_rejects_malformed_clients():
    for bad in ("{not json", '{"a": {"secret": "short", "scopes": ["query"]}}'):
        parse_clients.cache_clear()
        with pytest.raises(AuthConfigError):
            validate_auth_settings(_settings(auth_clients=bad))


def test_startup_rejects_a_typo_in_a_scope():
    bad = '{"a": {"secret": "0123456789abcdef", "scopes": ["ingset"]}}'
    with pytest.raises(AuthConfigError, match="unknown scopes"):
        validate_auth_settings(_settings(auth_clients=bad))


def test_settings_without_auth_config_still_load():
    # alembic and scripts load Settings without a JWT secret; that must work.
    assert _settings(jwt_secret=None).jwt_secret is None


# ------------------------------------------------------------------ over HTTP


async def test_v1_requires_a_token(anon_client):
    for method, path, body in (
        ("post", "/v1/query", {"question": "x"}),
        ("post", "/v1/search", {"query": "x"}),
        ("post", "/v1/ingest", {"content": "x"}),
    ):
        response = await getattr(anon_client, method)(path, json=body)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert response.headers["x-request-id"]


async def test_bad_token_gets_a_generic_401(anon_client):
    response = await anon_client.post(
        "/v1/query", json={"question": "x"}, headers=bearer("garbage")
    )
    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]
    # Nothing about *why* leaks to the caller.
    assert "signature" not in response.text.lower()
    assert "expired" not in response.text.lower()


async def test_valid_token_reaches_the_endpoint(client):
    # 422 = past auth and rate limiting, rejected by body validation.
    response = await client.post("/v1/query", json={"question": ""})
    assert response.status_code == 422


async def test_scope_is_enforced_per_route(anon_client):
    reader = bearer(make_token("test-reader", ["query"]))

    # query scope: allowed on query/search (422 = got past the scope check)...
    for path, body in (("/v1/query", {"question": ""}), ("/v1/search", {"query": ""})):
        response = await anon_client.post(path, json=body, headers=reader)
        assert response.status_code == 422, path

    # ...but not on ingest, even with an otherwise valid body.
    response = await anon_client.post(
        "/v1/ingest", json={"content": "new chunk"}, headers=reader
    )
    assert response.status_code == 403
    assert "ingest" in response.json()["detail"]


async def test_health_needs_no_token(anon_client):
    assert (await anon_client.get("/health/live")).status_code == 200


async def test_rate_limit_identity_is_the_token_subject(anon_client):
    # After authenticate() ran, the bucket must be keyed by the caller, not
    # their IP -- otherwise rotating IPs dodges the limit and one NAT shares it.
    from app.core.rate_limit import RateLimitResult
    from app.main import app

    seen = {}

    class _SpyLimiter:
        async def hit(self, identity, cost=1):
            seen["identity"] = identity
            return RateLimitResult(
                allowed=True, limit=20, remaining=19, retry_after_seconds=0
            )

        async def close(self):
            pass

    original = app.state.rate_limiter
    app.state.rate_limiter = _SpyLimiter()
    try:
        await anon_client.post(
            "/v1/query",
            json={"question": ""},
            headers=bearer(make_token("test-reader", ["query"])),
        )
    finally:
        app.state.rate_limiter = original

    assert seen["identity"] == "sub:test-reader"


# ------------------------------------------------------------------ /auth/token


async def test_token_endpoint_issues_a_usable_token(anon_client):
    response = await anon_client.post(
        "/auth/token",
        json={
            "client_id": "test-reader",
            "client_secret": TEST_CLIENTS["test-reader"]["secret"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["scope"] == "query"
    assert body["expires_in"] == get_settings().jwt_ttl_seconds

    # And the token it issued really works on /v1.
    follow_up = await anon_client.post(
        "/v1/query",
        json={"question": ""},
        headers=bearer(body["access_token"]),
    )
    assert follow_up.status_code == 422  # past auth


async def test_token_endpoint_rejects_bad_credentials_uniformly(anon_client):
    wrong_secret = await anon_client.post(
        "/auth/token", json={"client_id": "test-reader", "client_secret": "nope"}
    )
    unknown_id = await anon_client.post(
        "/auth/token", json={"client_id": "ghost", "client_secret": "nope"}
    )
    assert wrong_secret.status_code == unknown_id.status_code == 401
    assert wrong_secret.json() == unknown_id.json()  # can't tell which was wrong


async def test_token_scope_can_be_narrowed_but_not_widened(anon_client):
    creds = {
        "client_id": "test-admin",
        "client_secret": TEST_CLIENTS["test-admin"]["secret"],
    }
    narrowed = await anon_client.post("/auth/token", json={**creds, "scope": "query"})
    assert narrowed.status_code == 200
    assert narrowed.json()["scope"] == "query"

    reader = {
        "client_id": "test-reader",
        "client_secret": TEST_CLIENTS["test-reader"]["secret"],
    }
    widened = await anon_client.post("/auth/token", json={**reader, "scope": "ingest"})
    assert widened.status_code == 400


async def test_token_endpoint_is_404_when_auth_is_disabled(anon_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "auth_enabled", False)
    response = await anon_client.post(
        "/auth/token", json={"client_id": "x", "client_secret": "y"}
    )
    assert response.status_code == 404


async def test_auth_disabled_lets_requests_through(anon_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "auth_enabled", False)
    response = await anon_client.post("/v1/query", json={"question": ""})
    assert response.status_code == 422  # no token needed


def test_module_uses_the_configured_test_secret():
    # Guards the conftest ordering: if this fails, Settings was loaded before
    # the test env was applied and every other result here is suspect.
    assert get_settings().jwt_secret == TEST_JWT_SECRET
