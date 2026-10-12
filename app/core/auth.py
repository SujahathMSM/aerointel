"""JWT authentication (Phase 2, step 2).

Shape of the flow:
  1. A client posts its id + secret to POST /auth/token and gets a short-lived
     signed token back (OAuth2 "client credentials" style).
  2. It sends `Authorization: Bearer <token>` on every /v1 call.
  3. Every /v1 request verifies the signature, expiry, issuer and audience,
     and the route checks the token carries the scope it needs.

Design choices worth knowing:
  - Verification lives in ONE function (`decode_access_token`). In Phase 7
    Keycloak will issue the tokens instead; swapping that function for one
    that checks Keycloak's public keys is the whole change. Routes only ever
    see a `Principal`, never a raw token.
  - The accepted algorithm is pinned to the configured one (`algorithms=[...]`).
    Never "whatever the token header says" -- that is how `alg: none` and
    key-confusion attacks work.
  - `exp`, `iat`, `sub`, `iss` and `aud` are all REQUIRED, not just checked
    when present. A token without an expiry would be valid forever.
  - Scopes are coarse on purpose: `query` (ask + search) and `ingest` (write
    to the corpus). Ingest is the dangerous one -- it changes what every
    later answer is grounded in.
"""

import hmac
import json
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

import jwt
from pydantic import BaseModel, Field, ValidationError

from app.core.config import Settings

SCOPE_QUERY = "query"
SCOPE_INGEST = "ingest"
ALL_SCOPES = frozenset({SCOPE_QUERY, SCOPE_INGEST})

MIN_SECRET_CHARS = 32
# Seconds of clock drift tolerated between the issuer and the verifier.
_LEEWAY_SECONDS = 10
_REQUIRED_CLAIMS = ["exp", "iat", "sub", "iss", "aud"]


class AuthConfigError(RuntimeError):
    """Auth is enabled but misconfigured. Raised at startup, never per request."""


class TokenError(Exception):
    """A presented token is not acceptable. `reason` is for logs, not clients."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AuthClient(BaseModel):
    """One API client allowed to ask for tokens."""

    secret: str = Field(min_length=16)
    scopes: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class Principal:
    """Who is calling, as far as the rest of the app is concerned."""

    subject: str
    scopes: frozenset[str]
    authenticated: bool = True

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


# What requests run as when AUTH_ENABLED=false (local experiments only).
ANONYMOUS = Principal(subject="anonymous", scopes=ALL_SCOPES, authenticated=False)


@lru_cache(maxsize=8)
def parse_clients(raw: str) -> dict[str, AuthClient]:
    """Parse the AUTH_CLIENTS JSON: {"id": {"secret": "...", "scopes": [...]}}."""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        clients = {
            client_id: AuthClient.model_validate(entry)
            for client_id, entry in data.items()
        }
    except (json.JSONDecodeError, AttributeError, ValidationError) as exc:
        raise AuthConfigError(f"AUTH_CLIENTS is not valid: {exc}") from exc

    for client_id, client in clients.items():
        unknown = set(client.scopes) - ALL_SCOPES
        if unknown:
            raise AuthConfigError(
                f"AUTH_CLIENTS: client {client_id!r} has unknown scopes "
                f"{sorted(unknown)} (valid: {sorted(ALL_SCOPES)})"
            )
    return clients


def validate_auth_settings(settings: Settings) -> None:
    """Fail at startup, loudly, instead of 500-ing on the first request.

    Called from the API's lifespan -- deliberately NOT from Settings itself,
    so `alembic` and the scripts, which also load Settings, don't need a JWT
    secret just to run.
    """
    if not settings.auth_enabled:
        return
    secret = settings.jwt_secret or ""
    if len(secret) < MIN_SECRET_CHARS:
        raise AuthConfigError(
            f"JWT_SECRET must be set and at least {MIN_SECRET_CHARS} characters "
            "(or set AUTH_ENABLED=false for local experiments). "
            'Generate one with: python -c "import secrets; '
            'print(secrets.token_urlsafe(48))"'
        )
    parse_clients(settings.auth_clients)  # raises AuthConfigError if malformed


def verify_client(
    settings: Settings, client_id: str, client_secret: str
) -> AuthClient | None:
    """Return the client if id + secret match, else None.

    The secret is compared in constant time, and an unknown id still costs a
    comparison (against a dummy), so response timing doesn't reveal which
    client ids exist.
    """
    client = parse_clients(settings.auth_clients).get(client_id)
    expected = client.secret if client else "x" * MIN_SECRET_CHARS
    secret_ok = hmac.compare_digest(client_secret.encode(), expected.encode())
    return client if (client is not None and secret_ok) else None


def create_access_token(
    settings: Settings, subject: str, scopes: list[str]
) -> tuple[str, int]:
    """Sign a token for `subject`. Returns (token, lifetime_seconds)."""
    now = int(time.time())
    ttl = settings.jwt_ttl_seconds
    claims = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "sub": subject,
        "scope": " ".join(sorted(scopes)),  # OAuth2: space-delimited string
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,  # unique per token: audit + future revocation
    }
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, ttl


def decode_access_token(settings: Settings, token: str) -> Principal:
    """Verify a bearer token and return who it belongs to, or raise TokenError."""
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret or "",
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            leeway=_LEEWAY_SECONDS,
            options={"require": _REQUIRED_CLAIMS},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("expired") from exc
    except jwt.InvalidTokenError as exc:
        # Bad signature, wrong issuer/audience, missing claim, malformed...
        raise TokenError(f"invalid: {type(exc).__name__}") from exc

    subject = claims["sub"]
    scope_claim = claims.get("scope", "")
    if not isinstance(subject, str) or not subject:
        raise TokenError("invalid: empty subject")
    if not isinstance(scope_claim, str):
        raise TokenError("invalid: scope is not a string")
    return Principal(subject=subject, scopes=frozenset(scope_claim.split()))
