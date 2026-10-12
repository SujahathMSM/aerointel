# Must come first: configures the test auth environment before `app` loads.
from tests.auth_helpers import bearer, make_token  # isort: skip

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.main import app


@pytest.fixture
async def anon_client():
    """HTTP client with NO credentials, real lifespan running."""
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
async def client(anon_client):
    """The same client, signed in as an admin (query + ingest scopes).

    Most tests are about what /v1 does, not about logging in, so they get
    credentials by default; tests about auth itself use `anon_client`.
    """
    anon_client.headers.update(bearer(make_token()))
    return anon_client
