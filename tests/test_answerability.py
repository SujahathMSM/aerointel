"""Answerability: a question word that no stored chunk contains.

- question_terms and the spelling alternatives are pure.
- term_chunk_counts runs against a real Postgres (marked `integration`) with
  invented words, so a database full of real manuals cannot change the result.
- prepare_answer is checked with a fake session: the check runs only after the
  distance gate, "refuse" never reaches the model, "off" never queries.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base, DocumentChunk
from app.services import generation
from app.services.answerability import (
    _alternatives,
    check_answerability,
    judge,
    question_terms,
    term_chunk_counts,
)

integration = pytest.mark.integration

# ------------------------------------------------------------------ the terms


def test_everyday_question_words_are_not_tested():
    assert question_terms("How do I repair a cracked weld?") == [
        "repair",
        "cracked",
        "weld",
    ]


def test_model_numbers_are_tested_but_plain_numbers_are_not():
    terms = question_terms(
        "What is the weight of an A320 with 150 passengers on a G1000?"
    )
    assert "a320" in terms and "g1000" in terms
    assert "150" not in terms


def test_short_words_are_dropped_unless_they_mix_letters_and_digits():
    assert question_terms("fix the GE90 now") == ["ge90"]


def test_terms_are_lower_case_and_not_repeated():
    assert question_terms("Wire WIRE wire") == ["wire"]


def test_british_and_american_spellings_are_both_tried():
    assert "color" in _alternatives("colour")
    assert "colour" in _alternatives("color")
    assert "colored" in _alternatives("coloured")
    assert "recognize" in _alternatives("recognise")
    assert _alternatives("cable") == ["cable"]


def test_hour_is_not_turned_into_hor():
    assert _alternatives("hours") == ["hours"]


def test_judging_counts():
    verdict = judge({"airbus": 0, "cable": 12, "garmin": 0})
    assert verdict.unknown == ["airbus", "garmin"]
    assert not verdict.ok
    assert judge({"cable": 12}).ok


def test_a_word_in_too_few_chunks_is_unknown_when_asked_for_more():
    counts = {"helicopter": 1, "rotor": 6}
    assert judge(counts, min_chunks=1).ok
    assert judge(counts, min_chunks=2).unknown == ["helicopter"]


# ------------------------------------------------------------ against Postgres

SOURCE = "answerability-test"


@pytest.fixture
async def db_session():
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await _clean(session)
        session.add_all(
            DocumentChunk(content=text, metadata_={"source": SOURCE})
            for text in (
                "The flumoxiter housing is colored grey.",
                "Inspect each flumoxiter for wear.",
                "Torque the zibblewick clamp evenly.",
            )
        )
        await session.commit()
        yield session
        await session.rollback()
        await _clean(session)
    await engine.dispose()


async def _clean(session):
    await session.execute(
        delete(DocumentChunk).where(DocumentChunk.metadata_["source"].astext == SOURCE)
    )
    await session.commit()


@integration
async def test_counts_say_how_many_chunks_contain_each_word(db_session):
    counts = await term_chunk_counts(
        db_session, ["flumoxiter", "zibblewick", "quarbleglorp"]
    )
    assert counts == {"flumoxiter": 2, "zibblewick": 1, "quarbleglorp": 0}


@integration
async def test_inflected_forms_match(db_session):
    counts = await term_chunk_counts(db_session, ["flumoxiters", "inspecting"])
    assert counts["flumoxiters"] == 2
    assert counts["inspecting"] >= 1


@integration
async def test_the_other_spelling_matches(db_session):
    assert (await term_chunk_counts(db_session, ["coloured"]))["coloured"] >= 1
    assert (await term_chunk_counts(db_session, ["greyish"]))["greyish"] == 0


@integration
async def test_check_reports_only_the_unknown_words(db_session):
    found = await check_answerability(
        db_session, "How do I inspect the flumoxiter of the quarbleglorp?"
    )
    assert found.unknown == ["quarbleglorp"]
    assert found.terms["flumoxiter"] == 2


@integration
async def test_no_terms_is_not_a_failure(db_session):
    assert (await check_answerability(db_session, "how do i do it?")).ok


# ------------------------------------------------------ prepare_answer wiring


class Chunk:
    def __init__(self, cid, content="text"):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": "manual"}
        self.page = 1
        self.page_label = None


class FakeSession:
    def __init__(self):
        self.events: list[str] = []

    async def commit(self):
        self.events.append("commit")


class Verdict:
    def __init__(self, unknown):
        self.unknown = unknown


def _setup(monkeypatch, mode, unknown, min_unknown=1, distance=0.2):
    values = get_settings().model_copy(
        update={
            "answerability_mode": mode,
            "answerability_min_unknown": min_unknown,
            "context_neighbors": 0,
        }
    )
    calls: list[str] = []

    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        return [(Chunk(1, "some passage"), distance)]

    async def fake_check(session, question, min_chunks=1):
        calls.append(question)
        return Verdict(unknown)

    monkeypatch.setattr(generation, "get_settings", lambda: values)
    monkeypatch.setattr(generation, "retrieve", fake_retrieve)
    monkeypatch.setattr(generation, "check_answerability", fake_check)
    return calls


async def test_refuse_mode_refuses_without_building_a_prompt(monkeypatch):
    _setup(monkeypatch, "refuse", ["airbus"])
    session = FakeSession()
    prepared = await generation.prepare_answer(session, object(), "airbus weight?")
    assert prepared.refused is True
    assert prepared.refusal_reason == generation.REASON_UNKNOWN_TERMS
    assert prepared.unknown_terms == ["airbus"]
    assert prepared.prompt is None and prepared.system is None
    assert session.events == ["commit"]  # the connection is still released


async def test_refuse_mode_waits_for_enough_unknown_words(monkeypatch):
    _setup(monkeypatch, "refuse", ["judging"], min_unknown=2)
    prepared = await generation.prepare_answer(FakeSession(), object(), "q?")
    assert prepared.refused is False
    assert prepared.unknown_terms == ["judging"]


async def test_flag_mode_answers_and_reports_the_words(monkeypatch):
    _setup(monkeypatch, "flag", ["airbus"])
    prepared = await generation.prepare_answer(FakeSession(), object(), "q?")
    assert prepared.refused is False
    assert prepared.unknown_terms == ["airbus"]
    assert prepared.prompt


async def test_off_mode_never_looks(monkeypatch):
    calls = _setup(monkeypatch, "off", ["airbus"])
    prepared = await generation.prepare_answer(FakeSession(), object(), "q?")
    assert calls == []
    assert prepared.unknown_terms == []
    assert prepared.refused is False


async def test_a_question_the_distance_gate_refused_is_not_checked(monkeypatch):
    calls = _setup(monkeypatch, "refuse", ["airbus"], distance=0.95)
    prepared = await generation.prepare_answer(FakeSession(), object(), "q?")
    assert prepared.refused is True
    assert prepared.refusal_reason == generation.REASON_DISTANCE
    assert calls == []


async def test_a_refusal_for_unknown_words_names_them():
    text = generation.refusal_message(
        generation.REASON_UNKNOWN_TERMS, ["airbus", "garmin"], [], 0.48
    )
    assert "airbus, garmin" in text
