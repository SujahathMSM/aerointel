"""PDF document ingestion: the service, the upload endpoint, and page citations.

Three layers, cheapest first:
  - service tests with a fake session and fake Ollama (no services needed);
  - endpoint tests through the real app with the service replaced (need Redis,
    like every /v1 route);
  - integration tests against a real Postgres with pgvector (marked
    `integration`): dedup, page provenance, cascade delete, search by page.
"""

import hashlib

import pytest
import sqlalchemy as sa
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1 import router as router_module
from app.core.config import get_settings
from app.db.models import Base, Document, DocumentChunk
from app.services import documents
from app.services.documents import IngestedDocument, clean_filename, ingest_pdf
from app.services.generation import build_prompt
from app.services.pdf_text import PdfError
from app.services.retrieval import vector_search
from tests.auth_helpers import bearer, make_token
from tests.pdf_helpers import make_pdf

DIM = get_settings().embedding_dimensions


def sample_pdf(pages: int = 3, tag: str = "") -> bytes:
    """A small manual: running header/footer plus distinct body text per page."""
    return make_pdf(
        [
            [
                "9/8/98 AC 99.99-1A",
                f"Par 1-{i} Page 1-{i}",
                f"Section {i} {tag}: inspect the hydraulic reservoir sight glass before flight.",
                "If pressure falls below the limit, check the engine-driven pump for cavitation.",
                f"Reference {i}: replace the filter element when the indicator shows red.",
                "Record every finding in the aircraft maintenance log before release.",
            ]
            for i in range(1, pages + 1)
        ]
    )


def vector_for(text: str) -> list[float]:
    """Deterministic unit vector per text, so search results are predictable."""
    axis = int(hashlib.sha256(text.encode()).hexdigest(), 16) % DIM
    v = [0.0] * DIM
    v[axis] = 1.0
    return v


class FakeOllama:
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.events.append("embed")
        self.batches.append(list(texts))
        return [vector_for(t) for t in texts]


# ------------------------------------------------------------ service, fakes


class FakeSession:
    """Records what the service does to the database, in order."""

    def __init__(self, events: list[str] | None = None, flush_error=None):
        self.events = events if events is not None else []
        self.added: list = []
        self.flush_error = flush_error
        self._next_id = 41

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        self.events.append("flush")
        if self.flush_error is not None:
            raise self.flush_error
        for obj in self.added:
            if isinstance(obj, Document) and obj.id is None:
                obj.id = self._next_id

    async def commit(self):
        self.events.append("commit")

    async def rollback(self):
        self.events.append("rollback")


@pytest.fixture
def no_existing(monkeypatch):
    async def lookup(session, content_hash):
        session.events.append("lookup")

    monkeypatch.setattr(documents, "find_document_by_hash", lookup)


async def test_ingest_pdf_stores_document_and_page_bound_chunks(no_existing):
    events: list[str] = []
    session, ollama = FakeSession(events), FakeOllama(events)

    result = await ingest_pdf(session, ollama, sample_pdf(), "AC 99.99.pdf")

    document = next(o for o in session.added if isinstance(o, Document))
    chunks = [o for o in session.added if isinstance(o, DocumentChunk)]
    assert result.duplicate is False
    assert (result.document_id, result.pages, result.chunks) == (41, 3, len(chunks))
    assert result.title == "AC 99.99"
    assert document.content_hash == hashlib.sha256(sample_pdf()).hexdigest()
    assert document.byte_size == len(sample_pdf())
    assert document.page_count == 3

    assert {c.page for c in chunks} == {1, 2, 3}
    assert {(c.page, c.page_label) for c in chunks} == {
        (1, "1-1"),
        (2, "1-2"),
        (3, "1-3"),
    }
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == 41 for c in chunks)
    assert all(len(c.embedding) == DIM for c in chunks)
    assert all(c.embedding_model == get_settings().embedding_model for c in chunks)
    assert all(c.metadata_["source"] == "AC 99.99" for c in chunks)
    # running header and footer were stripped from the text that gets embedded
    assert not any("AC 99.99-1A" in c.content or "Par 1-" in c.content for c in chunks)


async def test_database_connection_is_released_before_the_slow_work(no_existing):
    """lookup -> commit (release) -> embed -> ... -> commit: never held while embedding."""
    events: list[str] = []
    await ingest_pdf(FakeSession(events), FakeOllama(events), sample_pdf(), "a.pdf")

    assert events[:3] == ["lookup", "commit", "embed"]
    assert events.index("commit") < events.index("embed")
    assert events[-1] == "commit"  # one final commit stores everything


async def test_chunks_are_embedded_in_batches(no_existing, monkeypatch):
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: get_settings().model_copy(update={"embed_batch_size": 2}),
    )
    ollama = FakeOllama()
    session = FakeSession()
    result = await ingest_pdf(session, ollama, sample_pdf(pages=4), "a.pdf")

    assert all(len(b) <= 2 for b in ollama.batches)
    assert sum(len(b) for b in ollama.batches) == result.chunks


async def test_progress_is_reported_after_each_batch(no_existing, monkeypatch):
    monkeypatch.setattr(
        documents,
        "get_settings",
        lambda: get_settings().model_copy(update={"embed_batch_size": 3}),
    )
    seen: list[tuple[int, int]] = []
    result = await ingest_pdf(
        FakeSession(),
        FakeOllama(),
        sample_pdf(pages=4),
        "a.pdf",
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen[-1] == (result.chunks, result.chunks)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


async def test_same_file_is_skipped_without_embedding(monkeypatch):
    existing = Document(
        id=7, filename="old.pdf", title="Old", content_hash="x", byte_size=1,
        page_count=9, chunk_count=30, ocr_pages=0,
    )  # fmt: skip

    async def lookup(session, content_hash):
        return existing

    monkeypatch.setattr(documents, "find_document_by_hash", lookup)
    events: list[str] = []
    ollama = FakeOllama(events)
    session = FakeSession(events)

    result = await ingest_pdf(session, ollama, sample_pdf(), "new-name.pdf")

    assert result == IngestedDocument(7, "Old", "old.pdf", 9, 30, 0, duplicate=True)
    assert ollama.batches == []
    assert session.added == []
    assert events == ["commit"]  # connection still released


async def test_a_concurrent_upload_of_the_same_file_is_treated_as_a_duplicate(
    monkeypatch,
):
    winner = Document(
        id=5, filename="w.pdf", title="W", content_hash="h", byte_size=1,
        page_count=3, chunk_count=3, ocr_pages=0,
    )  # fmt: skip
    lookups = iter([None, winner])  # first lookup: not there yet; after the race: there

    async def lookup(session, content_hash):
        return next(lookups)

    monkeypatch.setattr(documents, "find_document_by_hash", lookup)
    events: list[str] = []
    session = FakeSession(
        events, flush_error=IntegrityError("insert", {}, Exception("dup"))
    )

    result = await ingest_pdf(session, FakeOllama(events), sample_pdf(), "a.pdf")

    assert result.duplicate is True
    assert result.document_id == 5
    assert "rollback" in events
    assert not any(isinstance(o, DocumentChunk) for o in session.added)


async def test_unreadable_pdf_stores_nothing_and_embeds_nothing(no_existing):
    session, ollama = FakeSession(), FakeOllama()
    with pytest.raises(PdfError):
        await ingest_pdf(session, ollama, b"%PDF-1.4 broken", "bad.pdf")
    assert session.added == []
    assert ollama.batches == []


async def test_embedding_failure_stores_nothing(no_existing):
    class Failing(FakeOllama):
        async def embed(self, texts):
            raise RuntimeError("ollama is down")

    session = FakeSession()
    with pytest.raises(RuntimeError):
        await ingest_pdf(session, Failing(), sample_pdf(), "a.pdf")
    assert session.added == []  # nothing half-written


async def test_explicit_title_wins():
    result_title = "AC 43.13-1B Acceptable Methods"

    async def lookup(session, content_hash):
        return None

    import app.services.documents as module

    original = module.find_document_by_hash
    module.find_document_by_hash = lookup
    try:
        result = await ingest_pdf(
            FakeSession(), FakeOllama(), sample_pdf(), "x.pdf", result_title
        )
    finally:
        module.find_document_by_hash = original
    assert result.title == result_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("manual.pdf", "manual.pdf"),
        ("C:\\Users\\me\\Documents\\manual.pdf", "manual.pdf"),
        ("../../etc/passwd", "passwd"),
        ("/var/data/a b.pdf", "a b.pdf"),
        ("", "document.pdf"),
        ("x" * 400 + ".pdf", ("x" * 400 + ".pdf")[:255]),
    ],
)
def test_clean_filename_keeps_only_the_file_name(raw, expected):
    assert clean_filename(raw) == expected


# ----------------------------------------------------- citations in the prompt


class _Chunk:
    def __init__(self, content, source, page=None, page_label=None):
        self.content = content
        self.metadata_ = {"source": source}
        self.page = page
        self.page_label = page_label


def test_prompt_tells_the_model_which_page_each_passage_is_on():
    prompt = build_prompt(
        "q?",
        [
            (_Chunk("Alpha.", "AC 43.13-1B", page=121, page_label="3-3"), 0.1),
            (_Chunk("Beta.", "AC 43.13-1B", page=5), 0.2),
            (_Chunk("Gamma.", "notes"), 0.3),
        ],
    )
    assert "[1] Source: AC 43.13-1B, page 3-3 (PDF page 121)\nAlpha." in prompt
    assert "[2] Source: AC 43.13-1B, PDF page 5\nBeta." in prompt
    assert "[3] Source: notes\nGamma." in prompt  # plain-text chunks unchanged


# ------------------------------------------------------------- the endpoint


class FakeIngest:
    def __init__(self):
        self.calls = []
        self.error: Exception | None = None

    async def __call__(
        self, session, ollama, data, filename, title=None, progress=None
    ):
        self.calls.append((data, filename, title))
        if self.error:
            raise self.error
        return IngestedDocument(
            3, title or "Manual", filename, 12, 40, 0, duplicate=False
        )


@pytest.fixture
def fake_ingest(monkeypatch):
    fake = FakeIngest()
    monkeypatch.setattr(router_module, "ingest_pdf", fake)
    return fake


def _upload(
    client, content=b"%PDF-1.4 fake", name="manual.pdf", data=None, headers=None
):
    return client.post(
        "/v1/documents",
        files={"file": (name, content, "application/pdf")},
        data=data or {},
        headers=headers,
    )


@pytest.mark.redis
async def test_upload_returns_the_document_summary(client, fake_ingest):
    response = await _upload(client, data={"title": "My Manual"})
    assert response.status_code == 200
    assert response.json() == {
        "document_id": 3, "title": "My Manual", "filename": "manual.pdf",
        "pages": 12, "chunks": 40, "ocr_pages": 0, "duplicate": False,
    }  # fmt: skip
    assert fake_ingest.calls == [(b"%PDF-1.4 fake", "manual.pdf", "My Manual")]


@pytest.mark.redis
async def test_upload_rejects_non_pdf_files(client, fake_ingest):
    response = await _upload(client, content=b"just text", name="notes.pdf")
    assert response.status_code == 415
    assert fake_ingest.calls == []


@pytest.mark.redis
async def test_upload_rejects_an_empty_file(client, fake_ingest):
    response = await _upload(client, content=b"")
    assert response.status_code == 400
    assert fake_ingest.calls == []


@pytest.mark.redis
async def test_upload_rejects_files_over_the_size_limit(
    client, fake_ingest, monkeypatch
):
    monkeypatch.setattr(
        router_module,
        "get_settings",
        lambda: get_settings().model_copy(update={"max_upload_mb": 1}),
    )
    response = await _upload(client, content=b"%PDF-" + b"0" * (2 * 1024 * 1024))
    assert response.status_code == 413
    assert fake_ingest.calls == []


@pytest.mark.redis
async def test_unreadable_pdf_is_a_422_with_the_reason(client, fake_ingest):
    fake_ingest.error = PdfError("The PDF is password-protected.")
    response = await _upload(client)
    assert response.status_code == 422
    assert response.json()["detail"] == "The PDF is password-protected."


@pytest.mark.redis
async def test_upload_needs_a_token_and_the_ingest_scope(anon_client, fake_ingest):
    assert (await _upload(anon_client)).status_code == 401
    reader = bearer(make_token(subject="reader", scopes=("query",)))
    assert (await _upload(anon_client, headers=reader)).status_code == 403
    assert fake_ingest.calls == []


@pytest.mark.redis
async def test_search_results_carry_document_and_page(client, monkeypatch):
    chunk = _Chunk("Check the filter.", "AC 43.13-1B", page=121, page_label="3-3")
    chunk.id = 9
    chunk.document_id = 4

    async def fake_retrieve(session, ollama, query, top_k=None, **kwargs):
        return [(chunk, 0.12)]

    monkeypatch.setattr(router_module, "retrieve", fake_retrieve)
    response = await client.post("/v1/search", json={"query": "filter"})
    (hit,) = response.json()["results"]
    assert (hit["document_id"], hit["page"], hit["page_label"]) == (4, 121, "3-3")
    assert hit["source"] == "AC 43.13-1B"


# ---------------------------------------------- integration: a real Postgres

integration = pytest.mark.integration
MARKER_TITLE = "documents_test_manual"


@pytest.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(Document).where(Document.title == MARKER_TITLE))
        await session.commit()
        yield session
        await session.rollback()
        await session.execute(delete(Document).where(Document.title == MARKER_TITLE))
        await session.commit()
    await engine.dispose()


@integration
async def test_real_database_ingest_then_duplicate_then_cascade(db_session):
    pdf = sample_pdf(pages=3, tag="integration")
    ollama = FakeOllama()

    first = await ingest_pdf(db_session, ollama, pdf, "m.pdf", MARKER_TITLE)
    embedded = sum(len(b) for b in ollama.batches)
    second = await ingest_pdf(db_session, ollama, pdf, "renamed.pdf", "Another title")

    assert first.duplicate is False and second.duplicate is True
    assert second.document_id == first.document_id
    assert second.title == MARKER_TITLE  # the original record is returned
    assert sum(len(b) for b in ollama.batches) == embedded  # nothing re-embedded

    count = (
        await db_session.execute(
            select(sa.func.count())
            .select_from(Document)
            .where(Document.title == MARKER_TITLE)
        )
    ).scalar_one()
    assert count == 1

    rows = (
        (
            await db_session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == first.document_id)
                .order_by(DocumentChunk.chunk_index)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == first.chunks
    assert [r.chunk_index for r in rows] == list(range(len(rows)))
    assert {(r.page, r.page_label) for r in rows} == {
        (1, "1-1"),
        (2, "1-2"),
        (3, "1-3"),
    }
    assert all(r.embedding is not None for r in rows)

    # deleting the document removes its chunks (ON DELETE CASCADE)
    await db_session.execute(delete(Document).where(Document.id == first.document_id))
    await db_session.commit()
    left = (
        await db_session.execute(
            select(sa.func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == first.document_id)
        )
    ).scalar_one()
    assert left == 0


@integration
async def test_real_database_search_returns_the_page(db_session):
    pdf = sample_pdf(pages=3, tag="searchable")
    ollama = FakeOllama()
    result = await ingest_pdf(db_session, ollama, pdf, "m.pdf", MARKER_TITLE)

    chunks = (
        (
            await db_session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == result.document_id
                )
            )
        )
        .scalars()
        .all()
    )
    target = next(c for c in chunks if c.page == 2)

    hits = await vector_search(db_session, vector_for(target.content), top_k=3)
    top, distance = hits[0]
    assert top.id == target.id and distance < 1e-6
    assert (top.page, top.page_label, top.document_id) == (2, "1-2", result.document_id)
    assert "page 1-2 (PDF page 2)" in build_prompt("q?", [(top, distance)])
