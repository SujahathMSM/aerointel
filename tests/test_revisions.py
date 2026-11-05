"""Revision awareness: every answer names the issue of the document it quotes.

Headers are shaped like AC 43.13-1B's: unchanged pages keep the original issue
("9/8/98 AC 43.13-1B"), changed pages carry the change's ("9/27/01 AC 43.13-1B
CHG 1"), and the footer holds "Par 7-89 Page 7-39".
"""

from datetime import UTC, date, datetime

import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, Document
from app.main import app
from app.services import generation
from app.services.pdf_text import (
    document_key,
    document_revision,
    extract_pages,
    header_revision,
    latest_date,
    running_headers,
)
from app.services.revisions import find_source_revisions, source_revisions
from app.services.verification import verify_answer
from tests.pdf_helpers import make_pdf

ORIGINAL = "9/8/98 AC 43.13-1B"
CHANGED = "9/27/01 AC 43.13-1B CHG 1"

# --------------------------------------------------------------- headers


@pytest.mark.parametrize(
    ("header", "revision", "issued"),
    [
        (CHANGED, "AC 43.13-1B CHG 1", date(2001, 9, 27)),
        ("AC 43.13-1B CHG 1 9/27/01", "AC 43.13-1B CHG 1", date(2001, 9, 27)),
        (ORIGINAL, "AC 43.13-1B", date(1998, 9, 8)),
        (
            "9/27/01 AC 43.13-1B CHG 1 Par 3-2 Page 3-3",
            "AC 43.13-1B CHG 1",
            date(2001, 9, 27),
        ),
        ("12/1/2011 FAA-H-8083-30B", "FAA-H-8083-30B", date(2011, 12, 1)),
        ("Page 7-39", None, None),
        ("7-39", None, None),
        ("Par 7-89 Page 7-39", None, None),
    ],
)
def test_header_revision_and_issue(header, revision, issued):
    assert header_revision(header) == revision
    assert latest_date(header) == issued


def test_a_header_with_two_dates_gives_the_latest_issue():
    assert latest_date("AC 43.13-1B 9/8/98 CHG 1 9/27/01") == date(2001, 9, 27)


def test_two_digit_years_are_widened_around_1950():
    assert latest_date("1/2/49") == date(2049, 1, 2)
    assert latest_date("1/2/50") == date(1950, 1, 2)
    assert latest_date("13/45/01") is None  # not a date


def test_the_latest_dated_header_names_the_revision():
    headers = [ORIGINAL, ORIGINAL, CHANGED, ORIGINAL, None]
    assert document_revision(headers) == ("AC 43.13-1B CHG 1", date(2001, 9, 27))


def test_without_dates_the_most_common_header_then_the_title():
    assert document_revision(["Manual X", "Manual X", "Manual Y"])[0] == "Manual X"
    assert document_revision([None, "Page 3"], fallback="My Title") == (
        "My Title",
        None,
    )


def test_a_given_revision_always_wins():
    assert document_revision([CHANGED], given="AC 43.13-1B CHG 2") == (
        "AC 43.13-1B CHG 2",
        date(2001, 9, 27),
    )


def test_document_key_drops_the_change_marker():
    assert document_key("AC 43.13-1B CHG 1") == "AC 43.13-1B"
    assert document_key("AC 43.13-1B (Change 1)") == "AC 43.13-1B"
    assert document_key("AC 43.13-1B") == "AC 43.13-1B"
    assert document_key(None) is None


def _page(header, footer, body="Inspect the part for cracks and corrosion damage."):
    return [
        header,
        body,
        "Record the finding in the log book.",
        "Check torque.",
        footer,
    ]


def test_running_headers_take_a_dated_line_even_when_it_is_rare():
    # 1 changed page in 8: too rare to count as a repeated header, still its issue
    pages = [_page(ORIGINAL, f"Par 1-{i} Page 1-{i}") for i in range(1, 8)]
    pages.insert(3, _page(CHANGED, "Par 1-x Page 1-x"))
    headers = running_headers(pages)
    assert headers[3] == CHANGED
    assert headers[0] == ORIGINAL


def test_a_page_number_line_is_never_the_header():
    pages = [
        [f"Page 1-{i}", f"Body {i} of the page.", f"More text {i}.", str(i)]
        for i in range(1, 5)
    ]
    assert running_headers(pages) == [None] * 4


def test_a_mixed_issue_pdf_end_to_end():
    pdf = make_pdf(
        [
            [header, f"Par 2-{i} Page 2-{i}", f"Paragraph {i}: inspect the fitting."]
            + ["Replace any part that shows cracks or heavy corrosion."] * 3
            for i, header in enumerate([ORIGINAL, CHANGED, ORIGINAL, ORIGINAL], 1)
        ]
    )
    pages = extract_pages(pdf)
    assert [latest_date(p.header) for p in pages] == [
        date(1998, 9, 8),
        date(2001, 9, 27),
        date(1998, 9, 8),
        date(1998, 9, 8),
    ]
    assert "AC 43.13-1B" not in pages[0].text  # header still stripped from the text
    assert document_revision([p.header for p in pages])[0] == "AC 43.13-1B CHG 1"


# ---------------------------------------------------------- the source line


class Chunk:
    def __init__(self, content="Torque to 20-25 in-lb.", meta=None, page=394):
        self.id = 1
        self.content = content
        self.metadata_ = {"source": "AC_43.13-1B_CHG_1", **(meta or {})}
        self.page = page
        self.page_label = "7-39"
        self.document_id = None


REVISED = {"revision": "AC 43.13-1B CHG 1", "page_issued": "2001-09-27"}


def test_the_source_line_names_the_revision_and_the_page_issue():
    block = generation.context_block(Chunk(meta=REVISED))
    assert block.startswith(
        "Source: AC 43.13-1B CHG 1, page 7-39 (issued 2001-09-27, PDF page 394)\n"
    )


def test_chunks_stored_before_revisions_keep_the_old_source_line():
    assert generation.context_block(Chunk()).startswith(
        "Source: AC_43.13-1B_CHG_1, page 7-39 (PDF page 394)\n"
    )


def test_the_issue_date_is_not_part_of_the_verified_text():
    verified = generation.context_block(Chunk(meta=REVISED), for_verification=True)
    assert "2001-09-27" not in verified and "page 7-39" in verified


@pytest.mark.parametrize("wrong", ["27", "9"])
def test_a_wrong_figure_matching_the_issue_date_is_still_flagged(wrong, monkeypatch):
    values = get_settings().model_copy(update={"verify_mode": "flag"})
    monkeypatch.setattr(generation, "get_settings", lambda: values)
    chunk = Chunk(meta=REVISED)
    prepared = generation.PreparedAnswer(
        question="What torque?", refused=False, chunks=[(chunk, 0.2)]
    )
    result = generation.verify_prepared(prepared, f"Torque to {wrong} in-lb [1].")
    assert result is not None and f"{wrong} in-lb" in result.unsupported
    # and the date really would have covered it, had it been verified
    covered = verify_answer(
        f"Torque to {wrong} in-lb [1].",
        [generation.context_block(chunk)],
        "What torque?",
    )
    assert f"{wrong} in-lb" not in covered.unsupported


class _Llm:
    model_name = "fake-model"


def _prepared(chunk):
    return generation.PreparedAnswer(
        question="q?", refused=False, chunks=[(chunk, 0.2)], system="sys", prompt="p"
    )


def test_the_cache_bucket_changes_with_the_source_line():
    before = generation.cache_bucket(_prepared(Chunk()), _Llm())
    after = generation.cache_bucket(_prepared(Chunk(meta=REVISED)), _Llm())
    assert before != after  # an answer written without the revision is not reused


def test_the_cache_bucket_changes_with_the_prompt_layout(monkeypatch):
    before = generation.cache_bucket(_prepared(Chunk()), _Llm())
    original = generation.build_prompt

    def reworded(question, chunks):
        return original(question, chunks).replace("QUESTION:", "Q:")

    monkeypatch.setattr(generation, "build_prompt", reworded)
    assert generation.cache_bucket(_prepared(Chunk()), _Llm()) != before


# --------------------------------------------------------- newer revisions


def _doc(i, revision, when=None, key="AC 43.13-1B", superseded=False):
    return Document(
        id=i,
        title="AC_43.13-1B",
        revision=revision,
        revision_date=when,
        document_key=key,
        superseded=superseded,
        created_at=datetime(2026, 10, 16, tzinfo=UTC),
    )


OLD = _doc(1, "AC 43.13-1B", date(1998, 9, 8))
NEW = _doc(2, "AC 43.13-1B CHG 1", date(2001, 9, 27))


def test_a_newer_stored_revision_is_pointed_out():
    (old,) = source_revisions([OLD], [OLD, NEW])
    assert old.newer_revision == "AC 43.13-1B CHG 1" and old.newer_document_id == 2
    assert "A newer revision is stored: AC 43.13-1B CHG 1 (2001-09-27)" in old.note


def test_the_newest_revision_asks_to_be_confirmed_as_current():
    (new,) = source_revisions([NEW], [OLD, NEW])
    assert new.newer_revision is None
    assert new.note == (
        "From AC 43.13-1B CHG 1, as stored on 2026-10-16; "
        "confirm it is the current revision before use."
    )


def test_a_superseded_revision_says_so():
    marked = _doc(3, "AC 43.13-1B CHG 1", date(2001, 9, 27), superseded=True)
    (s,) = source_revisions([marked], [marked])
    assert s.superseded and "marked superseded" in s.note


def test_other_documents_are_not_newer_revisions():
    other = _doc(4, "AC 65-15A", date(2010, 1, 1), key="AC 65-15A")
    (old,) = source_revisions([OLD], [OLD, other])
    assert old.newer_revision is None


def test_without_dates_a_later_upload_of_another_revision_is_newer():
    a = _doc(5, "Manual Rev A")
    b = _doc(6, "Manual Rev B")
    (first,) = source_revisions([a], [a, b])
    assert first.newer_revision == "Manual Rev B"


# ------------------------------------------------------ real Postgres + API

TITLE = "revisions_test_manual"


@pytest.fixture
async def db_session():
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(Document).where(Document.title == TITLE))
        await session.commit()
        yield session
        await session.rollback()
        await session.execute(delete(Document).where(Document.title == TITLE))
        await session.commit()
    await engine.dispose()


def _stored(revision, when, n):
    return Document(
        filename="ac.pdf",
        title=TITLE,
        content_hash=f"revisions-test-{n}",
        byte_size=1,
        page_count=1,
        chunk_count=1,
        ocr_pages=0,
        revision=revision,
        revision_date=when,
        document_key=f"{TITLE} key",
    )


@pytest.mark.integration
async def test_real_database_finds_the_newer_revision(db_session):
    old = _stored("AC 43.13-1B", date(1998, 9, 8), 1)
    new = _stored("AC 43.13-1B CHG 1", date(2001, 9, 27), 2)
    db_session.add_all([old, new])
    await db_session.commit()

    class Read:
        document_id = old.id

    (source,) = await find_source_revisions(db_session, [(Read(), 0.2)])
    assert source.newer_revision == "AC 43.13-1B CHG 1"
    assert source.newer_document_id == new.id
    assert source.stored_on is not None


class _ApiLlm:
    model_name = "fake-model"

    async def chat(self, system, user):
        return "Torque to 20-25 in-lb [1]."

    async def embed(self, texts):
        return [[1.0] for _ in texts]


@pytest.mark.redis
async def test_query_names_the_revision_per_citation_and_per_source(
    client, monkeypatch
):
    chunk = Chunk(meta=REVISED)
    chunk.document_id = 64

    async def fake_retrieve(session, ollama, question, top_k=None, **kwargs):
        return [(chunk, 0.2)]

    async def fake_sources(session, chunks):
        return source_revisions([_doc(64, "AC 43.13-1B CHG 1", date(2001, 9, 27))], [])

    async def no_warnings(session, chunks, lookback):
        return []

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "find_source_revisions", fake_sources)
    monkeypatch.setattr(generation, "find_warnings", no_warnings)
    original = app.state.ollama
    app.state.ollama = _ApiLlm()
    try:
        body = (await client.post("/v1/query", json={"question": "torque?"})).json()
    finally:
        app.state.ollama = original

    assert body["chunks"][0]["revision"] == "AC 43.13-1B CHG 1"
    assert body["chunks"][0]["issued"] == "2001-09-27"
    (source,) = body["sources"]
    assert source["revision"] == "AC 43.13-1B CHG 1"
    assert source["revision_date"] == "2001-09-27"
    assert "confirm it is the current revision" in source["note"]
