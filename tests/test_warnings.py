"""Safety labels (WARNING / CAUTION / NOTE) found by code, never dropped.

The passages are shaped like AC 43.13-1B's extracted text: one line per page,
numbered paragraphs ("6-180. MAGNESIUM"), list items ("c. Tread Cuts."),
captions ("FIGURE 9-3."). Unit tests pass neighbours in directly; the service,
endpoint and real-Postgres tests at the end go through the app.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1 import router as router_module
from app.core.config import get_settings
from app.db.models import Base, Document, DocumentChunk
from app.main import app
from app.services import generation
from app.services.warnings import (
    MAX_WARNING_CHARS,
    SOURCE_BLOCK,
    SOURCE_PARAGRAPH_START,
    extract_warnings,
    find_warnings,
    join_chunks,
    mark_cited,
    warnings_for,
)
from tests.test_streaming import parse_sse

COPPER = (
    "6-179. COPPER AND COPPER ALLOYS. a. Copper corrosion produces a green film. "
    "WARNING: Brushing, sanding, and abrasive blasting of copper and copper alloys "
    "can be dangerous due to the creation of toxic airborne particles. Take "
    "necessary precautions to ensure safety. 6-180. MAGNESIUM AND ITS ALLOYS. "
    "Magnesium is the most chemically active metal used in aircraft."
)
TITANIUM = (
    "6-181. TITANIUM AND TITANIUM ALLOYS. a. Titanium is highly resistant. "
    "WARNING: Dry abrasive blasting of titanium can produce sparks. Make sure "
    "flammable vapors are not present. b. Titanium may discolour when heated."
)
TIRES = (
    "b. Tire Inspection. Inspect the tread for cuts. WARNING: Do not probe cuts "
    "or embedded foreign objects while tire is inflated. c. Tread Cuts. Cuts "
    "that expose the cord body require tire replacement."
)
TORQUE_TABLE = (
    "TABLE 7-1. TORQUE TABLE FOR STANDARD STEEL NUTS CAUTION THE FOLLOWING "
    "TORQUE VALUES ARE DERIVED FROM OIL FREE CADMIUM-PLATED THREADS. TORQUE "
    "VALUES IN INCH-POUNDS 8-36 12-15 10-24 20-25"
)
OXYGEN = (
    "9-45. OXYGEN SYSTEMS. CAUTION: Keep oil and grease away from oxygen fittings "
    "and equipment. NOTE: Oxygen cylinders are painted green. FIGURE 9-3. OXYGEN "
    "CYLINDER MARKINGS."
)


def _texts(text):
    return [(w.level, w.text) for w in extract_warnings(text)]


# ------------------------------------------------------------ where they end


def test_copper_warning_stops_at_the_next_numbered_paragraph():
    assert _texts(COPPER) == [
        (
            "WARNING",
            (
                "Brushing, sanding, and abrasive blasting of copper and copper alloys "
                "can be dangerous due to the creation of toxic airborne particles. "
                "Take necessary precautions to ensure safety."
            ),
        )
    ]


def test_titanium_warning_stops_at_the_next_list_item():
    assert _texts(TITANIUM) == [
        (
            "WARNING",
            (
                "Dry abrasive blasting of titanium can produce sparks. Make sure "
                "flammable vapors are not present."
            ),
        )
    ]


def test_tire_warning_stops_before_the_tread_cuts_item():
    assert _texts(TIRES) == [
        (
            "WARNING",
            "Do not probe cuts or embedded foreign objects while tire is inflated.",
        )
    ]


def test_a_table_banner_without_a_colon_keeps_one_sentence():
    assert _texts(TORQUE_TABLE) == [
        (
            "CAUTION",
            (
                "THE FOLLOWING TORQUE VALUES ARE DERIVED FROM OIL FREE "
                "CADMIUM-PLATED THREADS."
            ),
        )
    ]


def test_a_caution_and_a_note_stay_separate_and_the_note_stops_at_a_figure():
    assert _texts(OXYGEN) == [
        ("CAUTION", "Keep oil and grease away from oxygen fittings and equipment."),
        ("NOTE", "Oxygen cylinders are painted green."),
    ]


def test_section_heading_and_plural_label():
    text = "NOTES: Values are typical. SECTION 2. RIVETS"
    assert _texts(text) == [("NOTE", "Values are typical.")]


@pytest.mark.parametrize(
    "text",
    [
        "Set the Huck rivet gun to the pull length. No warning here.",
        "Refer to the note in the manufacturer's manual.",
        "See NOTE",
        "See NOTE 3 in the table below.",
        "Heed the WARNING above before starting.",
        "WARNING:",
        # only upper-case labels count, even at the start of a sentence
        "Clean the part. note that the tag is dated.",
        "Check the panel. Warning lights must work before flight.",
    ],
)
def test_no_false_warnings(text):
    assert extract_warnings(text) == []


def test_a_long_warning_is_cut_at_a_sentence_end():
    sentence = "Wear eye protection when grinding the part. "
    text = "WARNING: " + sentence * 30
    (found,) = extract_warnings(text)
    assert len(found.text) <= MAX_WARNING_CHARS
    assert found.text.endswith("part.")
    assert text[len("WARNING: ") :].startswith(found.text)  # verbatim


# --------------------------------------------------------- the answer's list


class Chunk:
    def __init__(self, content, doc=None, index=None, page=None, label=None, cid=0):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": "AC 43.13-1B"}
        self.document_id = doc
        self.chunk_index = index
        self.page = page
        self.page_label = label


def _read(*chunks):
    return [(c, 0.3) for c in chunks]


def test_most_severe_first_then_reading_order():
    warnings = warnings_for(_read(Chunk(OXYGEN), Chunk(COPPER), Chunk(TORQUE_TABLE)))
    assert [(w.level, w.block) for w in warnings] == [
        ("WARNING", 2),
        ("CAUTION", 1),
        ("CAUTION", 3),
        ("NOTE", 1),
    ]
    assert all(w.source == SOURCE_BLOCK for w in warnings)


def test_overlapping_chunks_give_one_warning():
    # Two chunks of the same page repeat the warning (chunk overlap).
    first = Chunk(TIRES)
    second = Chunk(TIRES[TIRES.index("WARNING") :])
    warnings = warnings_for(_read(first, second))
    assert len(warnings) == 1 and warnings[0].block == 1


def test_a_cut_off_copy_gives_way_to_the_complete_warning():
    cut = Chunk("Inspect the tread. WARNING: Do not probe cuts or embedded")
    whole = Chunk(TIRES)
    (warning,) = warnings_for(_read(cut, whole))
    assert warning.text.endswith("while tire is inflated.")
    assert warning.block == 1  # still where it was first seen


def test_cited_warnings_come_first_within_a_level():
    warnings = warnings_for(_read(Chunk(COPPER), Chunk(TITANIUM)))
    assert [w.block for w in warnings] == [1, 2]
    cited = mark_cited(warnings, [2])
    assert [(w.block, w.cited) for w in cited] == [(2, True), (1, False)]


def test_join_chunks_removes_the_repeated_overlap():
    a = Chunk("Alpha beta gamma delta. Epsilon zeta eta theta iota.")
    b = Chunk("Epsilon zeta eta theta iota. Kappa lambda mu.")
    text, pieces = join_chunks([a, b])
    assert (
        text == "Alpha beta gamma delta. Epsilon zeta eta theta iota. Kappa lambda mu."
    )
    assert text[pieces[1].start : pieces[1].end] == b.content


# ---------------------------------------------------- neighbours in the store

PARAGRAPH = [
    # index 10: the paragraph heading and its warning
    (
        "6-181. TITANIUM AND TITANIUM ALLOYS. WARNING: Dry abrasive blasting of "
        "titanium can produce sparks. Make sure flammable vapors are not present."
    ),
    # 11, 12: more of the same paragraph
    "a. Titanium is highly resistant to corrosion in most environments.",
    "b. Titanium may discolour when heated during welding or grinding.",
    # 13: the chunk that was retrieved
    "c. The coloured oxide film need not be removed unless it flakes.",
]


def _doc(texts, doc=64, first_index=10, page=349):
    return {
        (doc, first_index + i): Chunk(t, doc, first_index + i, page, f"6-{40 + i}")
        for i, t in enumerate(texts)
    }


def test_a_warning_at_the_paragraph_start_is_found_by_lookback():
    stored = _doc(PARAGRAPH)
    read = stored[(64, 13)]
    (warning,) = warnings_for(_read(read), stored, lookback=4)
    assert warning.source == SOURCE_PARAGRAPH_START
    assert warning.block is None  # not a block the model read: no [n]
    assert warning.page == 349 and warning.page_label == "6-40"
    assert warning.text.startswith("Dry abrasive blasting")


def test_lookback_is_limited_in_chunks():
    stored = _doc(PARAGRAPH)
    read = stored[(64, 13)]
    assert warnings_for(_read(read), stored, lookback=2) == []


def test_lookback_stops_at_the_paragraph_heading():
    texts = [
        PARAGRAPH[0],
        "6-182. NICKEL AND NICKEL ALLOYS. Nickel alloys resist heat.",
        "a. Inspect nickel parts for pitting near fasteners and welds.",
    ]
    stored = _doc(texts)
    read = stored[(64, 12)]
    assert warnings_for(_read(read), stored, lookback=4) == []


def test_lookback_stays_in_the_same_document():
    stored = _doc(PARAGRAPH[:1], doc=7, first_index=12) | _doc(
        PARAGRAPH[1:2], first_index=13
    )
    read = stored[(64, 13)]
    assert warnings_for(_read(read), stored, lookback=4) == []


def test_a_warning_cut_by_the_chunk_end_is_completed_from_the_next_chunk():
    first = "b. Tire Inspection. Inspect the tread for cuts. WARNING: Do not probe cuts"
    # the next chunk repeats the last sentence as overlap, then continues
    second = (
        "Inspect the tread for cuts. WARNING: Do not probe cuts or embedded foreign "
        "objects while tire is inflated. c. Tread Cuts."
    )
    stored = _doc([first, second], first_index=20)
    (warning,) = warnings_for(_read(stored[(64, 20)]), stored, lookback=4)
    assert warning.block == 1
    assert warning.text == (
        "Do not probe cuts or embedded foreign objects while tire is inflated."
    )


def test_a_warning_after_the_read_chunk_is_not_added():
    texts = [
        "a. Titanium is highly resistant to corrosion in most environments.",
        PARAGRAPH[0].split("6-181. TITANIUM AND TITANIUM ALLOYS. ")[1],
    ]
    stored = _doc(texts, first_index=30)
    assert warnings_for(_read(stored[(64, 30)]), stored, lookback=4) == []


def test_a_warning_in_a_read_block_beats_the_same_one_at_a_paragraph_start():
    stored = _doc(PARAGRAPH)
    # [1] is the chunk after the warning (lookback finds it), [2] holds it
    warnings = warnings_for(
        _read(stored[(64, 11)], stored[(64, 10)]), stored, lookback=4
    )
    (warning,) = warnings
    assert warning.source == SOURCE_BLOCK and warning.block == 2
    assert mark_cited(warnings, [1])[0].cited  # it leads into [1] too


# ------------------------------------------------- in the answer service


TITANIUM_HIT = Chunk(TITANIUM, cid=1, page=349, label="6-40")
COPPER_FAR = Chunk(COPPER, cid=2, page=348, label="6-39")


class _Session:
    async def commit(self):
        pass


class _Llm:
    model_name = "fake-model"

    def __init__(self, answer="Titanium may be blasted [1].", check="YES"):
        self.answer = answer
        self.check = check

    async def chat(self, system, user):
        return self.check if user.endswith("REPLY (YES or NO):") else self.answer

    def chat_stream(self, system, user):
        async def gen():
            yield self.answer

        return gen()

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def retrieved(monkeypatch):
    state = {"chunks": [(TITANIUM_HIT, 0.30)], "settings": {}}

    async def fake_retrieve(session, ollama, question, top_k=None, **kwargs):
        return state["chunks"]

    def settings():
        return get_settings().model_copy(update=state["settings"])

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "get_settings", settings)
    return state


async def test_the_answer_omits_the_warning_and_the_response_still_carries_it(
    retrieved,
):
    result = await generation.answer_question(_Session(), _Llm(), "blasting?")
    assert "spark" not in (result.answer or "")
    (warning,) = result.warnings
    assert warning.level == "WARNING" and "flammable vapors" in warning.text
    assert warning.block == 1 and warning.cited is True
    assert warning.page == 349 and warning.page_label == "6-40"


async def test_a_passage_beyond_the_reading_limit_adds_no_warning(retrieved):
    gate = get_settings().max_distance
    retrieved["chunks"] = [(TITANIUM_HIT, 0.30), (COPPER_FAR, gate + 0.05)]
    result = await generation.answer_question(_Session(), _Llm(), "blasting?")
    assert [w.text[:3] for w in result.warnings] == ["Dry"]  # no copper warning


async def test_a_refused_question_gets_no_warnings(retrieved):
    retrieved["chunks"] = [(TITANIUM_HIT, get_settings().max_distance + 0.1)]
    result = await generation.answer_question(_Session(), _Llm(), "blasting?")
    assert result.refused and result.warnings == []


async def test_an_answer_check_refusal_gets_no_warnings(retrieved):
    retrieved["settings"] = {"answer_check": "refuse"}
    result = await generation.answer_question(_Session(), _Llm(check="NO"), "blasting?")
    assert result.refused and result.warnings == []


async def test_an_answer_withheld_by_the_figure_check_keeps_its_warnings(retrieved):
    # The reader is told to read the passages instead: their safety text stays.
    retrieved["settings"] = {"verify_mode": "block"}
    llm = _Llm(answer="Blast at 90 psi [1].")
    result = await generation.answer_question(_Session(), llm, "blasting?")
    assert result.refused and result.refusal_reason == generation.REASON_UNVERIFIED
    assert [w.level for w in result.warnings] == ["WARNING"]


# ----------------------------------------------------------- the endpoints


@pytest.fixture
async def api(client, retrieved):
    original = app.state.ollama
    app.state.ollama = _Llm()
    yield client
    app.state.ollama = original


@pytest.mark.redis
async def test_query_endpoint_returns_warnings(api):
    body = (await api.post("/v1/query", json={"question": "blasting?"})).json()
    assert body["refused"] is False
    assert body["warnings"] == [
        {
            "level": "WARNING",
            "text": "Dry abrasive blasting of titanium can produce sparks. Make "
            "sure flammable vapors are not present.",
            "block": 1,
            "source": "block",
            "page": 349,
            "page_label": "6-40",
            "cited": True,
        }
    ]


@pytest.mark.redis
async def test_stream_meta_carries_warnings_before_any_token(api, monkeypatch):
    monkeypatch.setattr(router_module, "stream_answer", generation.stream_answer)
    events = parse_sse(
        (await api.post("/v1/query/stream", json={"question": "blasting?"})).text
    )
    name, meta = events[0]
    assert name == "meta"
    assert [w["level"] for w in meta["warnings"]] == ["WARNING"]
    assert meta["warnings"][0]["cited"] is False  # no answer yet


@pytest.mark.redis
async def test_refused_query_endpoint_has_no_warnings(api, retrieved):
    retrieved["chunks"] = [(TITANIUM_HIT, 0.95)]
    body = (await api.post("/v1/query", json={"question": "blasting?"})).json()
    assert body["refused"] is True and body["warnings"] == []


# ------------------------------------------------- integration: real Postgres


TITLE = "warnings_test_manual"


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


@pytest.mark.integration
async def test_real_database_lookback_finds_the_paragraph_warning(db_session):
    doc = Document(
        filename="w.pdf",
        title=TITLE,
        content_hash="warnings-test-hash",
        byte_size=1,
        page_count=1,
        chunk_count=len(PARAGRAPH),
        ocr_pages=0,
    )
    db_session.add(doc)
    await db_session.flush()
    rows = [
        DocumentChunk(
            content=text,
            metadata_={"source": TITLE},
            document_id=doc.id,
            chunk_index=i,
            page=349,
            page_label="6-40",
        )
        for i, text in enumerate(PARAGRAPH)
    ]
    db_session.add_all(rows)
    await db_session.commit()

    (warning,) = await find_warnings(db_session, [(rows[3], 0.3)], lookback=4)
    assert warning.source == SOURCE_PARAGRAPH_START and warning.page == 349
    assert await find_warnings(db_session, [(rows[3], 0.3)], lookback=2) == []
