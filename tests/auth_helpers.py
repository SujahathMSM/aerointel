"""Shared auth fixtures-by-function for the test suite.

Importing this module configures the test auth environment, so it must be
imported before anything that loads `app` (conftest.py does that first).
"""

import json
import os
import uuid

# Assigned (not setdefault) so a developer's real .env / shell values can
# never leak into the tests.
TEST_JWT_SECRET = "test-secret-" + "k" * 40
TEST_CLIENTS = {
    "test-admin": {"secret": "admin-secret-0123456789", "scopes": ["query", "ingest"]},
    "test-reader": {"secret": "reader-secret-0123456789", "scopes": ["query"]},
}
os.environ["AUTH_ENABLED"] = "true"
os.environ["JWT_SECRET"] = TEST_JWT_SECRET
os.environ["AUTH_CLIENTS"] = json.dumps(TEST_CLIENTS)

# The app's own rate limiter talks to the real Redis and keeps its buckets
# between runs. Without isolation, a few quick test runs would drain the
# caller's bucket and the next run would get spurious 429s. So: a private key
# prefix per test process, and a bucket far bigger than any test needs.
# (tests/test_rate_limit.py builds its own small limiters to test limiting.)
os.environ["RATE_LIMIT_KEY_PREFIX"] = f"test-app:{uuid.uuid4().hex}"
os.environ["RATE_LIMIT_CAPACITY"] = "1000"
# Same isolation for idempotency keys (tests reuse key strings like "k1").
os.environ["IDEMPOTENCY_KEY_PREFIX"] = f"test-idem:{uuid.uuid4().hex}"

# The trust-layer switches are read from the developer's .env too (they set
# ANSWERABILITY_MODE=flag to try it live), and a fake DB session cannot run the
# word check. Pin them to the shipped defaults; tests that exercise a mode pass
# it explicitly through model_copy(update=...).
os.environ["VERIFY_MODE"] = "flag"
os.environ["ANSWERABILITY_MODE"] = "off"
os.environ["ANSWERABILITY_MIN_CHUNKS"] = "1"
os.environ["ANSWERABILITY_MIN_UNKNOWN"] = "1"
# Same for chunking: a developer re-ingesting with CHUNK_BY_PARAGRAPH=true must
# not change what the ingest tests chunk. Tests that want it pass it explicitly.
os.environ["CHUNK_BY_PARAGRAPH"] = "false"

from app.core.auth import create_access_token
from app.core.config import get_settings


def make_token(subject: str = "test-admin", scopes=("query", "ingest")) -> str:
    """A valid access token, minted directly (no HTTP round trip)."""
    token, _ = create_access_token(get_settings(), subject, list(scopes))
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
