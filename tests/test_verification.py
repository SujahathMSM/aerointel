"""Number verification: an answer's figures must be in the passages it cites.

Pure-text checks (no database, no model). The passages are real sentences from
FAA AC 43.13-1B, so the extraction is tried on the formats a manual really uses:
"700-800 degrees F", "inch-pounds", ranges, fractions, "1,000".
"""

import pytest

from app.api.v1 import router as router_module
from app.core.config import Settings, get_settings
from app.services import generation
from app.services.generation import AnswerResult, PreparedAnswer
from app.services.verification import (
    Verification,
    numbers_in,
    parse_citations,
    verify_answer,
)

TITANIUM = (
    "6-181. TITANIUM AND TITANIUM ALLOYS. When titanium is heated, different "
    "oxides having different colors form on the surface. A blue oxide coating "
    "will form at 700–800 °F, a purple oxide at 800–950 °F, and a gray "
    "or black oxide at 1,000 °F or higher. These coatings are protective "
    "discolorations and should not be removed."
)
TORQUE = (
    "Tighten the nut to 20-25 inch-pounds, then secure it with 0.032-inch "
    "safety wire. The hole spacing is 3/8 in. and the clamp is AN960-10."
)


# ----------------------------------------------------------------- extraction


def values(text):
    return [(v, u) for v, u, _w in numbers_in(text)]


def test_decimals_fractions_and_thousands_are_normalised():
    assert values("0.032, .032 and 0.0320")[0] == values(".032")[0]
    assert values("1,000 lb") == [("1000", "lb")]
    assert values("3/8 in.") == [("3/8", "in")]


def test_a_unit_after_a_range_belongs_to_both_ends():
    assert values("20-25 inch-pounds") == [("20", "in-lb"), ("25", "in-lb")]
    assert values("3 to 5 psi") == [("3", "psi"), ("5", "psi")]


def test_unit_spellings_are_folded():
    assert values("2 inches")[0][1] == values("2 in.")[0][1] == "in"
    assert values("700 °F")[0][1] == values("700 degrees F")[0][1] == "°F"
    assert values("10 percent")[0][1] == values("10%")[0][1] == "%"


def test_a_preposition_after_a_number_is_not_a_unit():
    assert values("repeat 5 in the same order") == [("5", None)]


def test_citations_and_list_markers_are_not_numbers():
    assert values("1. First step [1]\n2. Second step [2, 3]") == []


def test_digits_inside_identifiers_are_ignored():
    assert values("use an AN960-10 washer on the A320") == [("10", None)]


def test_citation_forms():
    assert parse_citations("see [1] and [2, 3] and [4-6]") == [1, 2, 3, 4, 5, 6]
    assert parse_citations("no citation here") == []
    assert parse_citations("[2] then [2] again") == [2]


# --------------------------------------------------------------- verification


def test_a_faithful_answer_passes():
    answer = (
        "A blue oxide forms at 700-800 °F and a gray or black oxide at "
        "1,000 °F or higher [1]."
    )
    result = verify_answer(answer, [TITANIUM])
    assert result.ok
    assert result.checked == 3
    assert result.cited == [1]


@pytest.mark.parametrize(
    ("wrong", "figure"),
    [
        ("A blue oxide forms at 700-900 °F [1].", "900 °F"),
        ("A blue oxide forms at 70-800 °F [1].", "70"),
        ("A gray or black oxide forms at 1,100 °F [1].", "1,100 °F"),
    ],
)
def test_one_changed_digit_is_caught(wrong, figure):
    """The mutation check: alter one figure of a correct answer and it must fail."""
    result = verify_answer(wrong, [TITANIUM])
    assert not result.ok
    assert figure in result.unsupported


def test_a_wrong_unit_on_a_right_number_is_caught():
    result = verify_answer("Tighten to 20-25 foot-pounds [1].", [TORQUE])
    assert not result.ok
    assert result.unsupported == []
    assert any("foot-pounds" in w for w in result.unit_mismatch)


def test_a_number_without_a_unit_in_the_passage_is_not_a_unit_mismatch():
    # a table cell: "25" with no unit printed beside it
    assert verify_answer("The value is 25 psi [1].", ["Row 3: 25"]).ok


def test_the_wire_size_can_be_written_either_way():
    assert verify_answer("Use .032 in. safety wire [1].", [TORQUE]).ok


def test_a_figure_the_question_already_contains_may_be_repeated():
    answer = "A 0.040-inch wire is not what the passage specifies [1]."
    assert not verify_answer(answer, [TORQUE]).ok
    assert verify_answer(answer, [TORQUE], question="Is 0.040 inch wire ok?").ok


def test_only_the_cited_passage_counts():
    other = "The proof load is 4,000 lb."
    answer = "The proof load is 4,000 lb [2]."
    assert verify_answer(answer, [TITANIUM, other]).ok
    assert not verify_answer(answer, [other, TITANIUM]).ok  # [2] is the wrong one


def test_without_citations_every_passage_counts():
    assert verify_answer("It forms at 800 °F.", [TITANIUM, TORQUE]).ok


def test_a_citation_to_a_missing_block_is_reported():
    result = verify_answer("See [3].", [TITANIUM])
    assert result.bad_citations == [3]
    assert not result.ok


def test_an_answer_with_no_figures_has_nothing_to_check():
    result = verify_answer("The coatings should not be removed [1].", [TITANIUM])
    assert result.ok
    assert result.checked == 0


def test_the_same_wrong_figure_is_listed_once():
    result = verify_answer("It is 99 psi, yes 99 psi [1].", [TORQUE])
    assert result.unsupported == ["99 psi"]


# ------------------------------------------------- wiring into answer_question


class FakeOllama:
    def __init__(self, text):
        self.text = text

    async def chat(self, system, prompt):
        return self.text


class _Chunk:
    def __init__(self, content):
        self.id = 1
        self.content = content
        self.metadata_ = {"source": "AC"}
        self.page = 3
        self.page_label = "6-43"


def _prepared(content=TITANIUM):
    return PreparedAnswer(
        question="At what temperature does titanium turn blue?",
        refused=False,
        chunks=[(_Chunk(content), 0.2)],
        system="sys",
        prompt="prompt",
        prompt_version="v1",
    )


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value):
        values_ = get_settings().model_copy(update={"verify_mode": value})
        monkeypatch.setattr(generation, "get_settings", lambda: values_)
        monkeypatch.setattr(router_module, "get_settings", lambda: values_)

    return set_mode


async def _ask(monkeypatch, text, prepared=None):
    prepared = prepared or _prepared()

    async def fake_prepare(session, ollama, question, top_k=None):
        return prepared

    monkeypatch.setattr(generation, "prepare_answer", fake_prepare)
    return await generation.answer_question(None, FakeOllama(text), prepared.question)


async def test_flag_mode_returns_the_answer_with_the_findings(monkeypatch, mode):
    mode("flag")
    result = await _ask(monkeypatch, "Blue forms at 900 °F [1].")
    assert result.refused is False
    assert result.answer
    assert result.verification is not None
    assert result.verification.unsupported == ["900 °F"]


async def test_block_mode_withholds_an_answer_with_an_unsupported_figure(
    monkeypatch, mode
):
    mode("block")
    result = await _ask(monkeypatch, "Blue forms at 900 °F [1].")
    assert result.refused is True
    assert result.answer is None
    assert result.refusal_reason == generation.REASON_UNVERIFIED
    assert result.verification is not None
    assert not result.verification.ok


async def test_block_mode_lets_a_verified_answer_through(monkeypatch, mode):
    mode("block")
    result = await _ask(monkeypatch, "Blue forms at 700-800 °F [1].")
    assert result.refused is False
    assert result.answer


async def test_off_mode_skips_the_check(monkeypatch, mode):
    mode("off")
    result = await _ask(monkeypatch, "Blue forms at 900 °F [1].")
    assert result.verification is None
    assert result.answer


def test_the_withheld_message_names_the_figures():
    text = generation.refusal_message(
        generation.REASON_UNVERIFIED, [], ["900 °F"], 0.48
    )
    assert "900 °F" in text and "withheld" in text


def test_the_settings_default_to_flagging_and_not_refusing():
    s = Settings(_env_file=None, postgres_password="x")
    assert s.verify_mode == "flag"
    assert s.answerability_mode == "off"
    assert s.answerability_min_chunks == 1
    assert s.answerability_min_unknown == 1


# --------------------------------------------------------------- the endpoint


@pytest.mark.redis
async def test_query_response_carries_the_verification(client, monkeypatch):
    verification = Verification(checked=2, cited=[1], unsupported=["900 °F"])

    async def fake_answer(session, ollama, question, top_k=None):
        return AnswerResult(
            question=question,
            refused=False,
            answer="Blue forms at 900 °F [1].",
            chunks=[(_Chunk(TITANIUM), 0.2)],
            prompt_version="v1",
            unknown_terms=["zorptangle"],
            verification=verification,
        )

    monkeypatch.setattr(router_module, "answer_question", fake_answer)
    body = (await client.post("/v1/query", json={"question": "q?"})).json()
    assert body["refused"] is False
    assert body["unknown_terms"] == ["zorptangle"]
    assert body["verification"] == {
        "ok": False,
        "checked": 2,
        "cited": [1],
        "unsupported": ["900 °F"],
        "unit_mismatch": [],
        "bad_citations": [],
        "found_in_uncited": {},
        "blocked": False,
    }


@pytest.mark.redis
async def test_a_withheld_answer_is_a_refusal_with_its_reason(client, monkeypatch):
    verification = Verification(checked=1, unsupported=["900 °F"])

    async def fake_answer(session, ollama, question, top_k=None):
        return AnswerResult(
            question=question,
            refused=True,
            chunks=[(_Chunk(TITANIUM), 0.2)],
            refusal_reason=generation.REASON_UNVERIFIED,
            verification=verification,
        )

    monkeypatch.setattr(router_module, "answer_question", fake_answer)
    body = (await client.post("/v1/query", json={"question": "q?"})).json()
    assert body["refused"] is True
    assert body["answer"] is None
    assert body["refusal_reason"] == "unverified_numbers"
    assert "900 °F" in body["refusal_message"]
    assert body["verification"]["blocked"] is True


# --------------------------------------- page references from the source line

TORQUE_TABLE = (
    "TABLE 7-1. TORQUE VALUES IN INCH-POUNDS FOR TIGHTENING NUTS. "
    "8-36 nut: 12-15 inch-pounds."
)


def _pdf_chunk(content, page, label):
    chunk = _Chunk(content)
    chunk.metadata_ = {"source": "AC_43.13-1B_CHG_1"}
    chunk.page = page
    chunk.page_label = label
    return chunk


def test_the_source_line_is_part_of_the_checked_block():
    block = generation.context_block(_pdf_chunk(TORQUE_TABLE, 364, "7-9"))
    assert block.startswith("Source: AC_43.13-1B_CHG_1, page 7-9 (PDF page 364)\n")
    assert block.endswith(TORQUE_TABLE)
    # The prompt shows the model exactly what the check reads.
    prompt = generation.build_prompt("q", [(_pdf_chunk(TORQUE_TABLE, 364, "7-9"), 0.3)])
    assert f"[1] {block}" in prompt


async def test_a_page_copied_from_the_source_line_is_not_flagged(monkeypatch, mode):
    # Live run 2026-10-17, default-torque-table: the answer named "PDF page 364"
    # from the source line and the check flagged 364 as an unsupported figure.
    mode("flag")
    prepared = _prepared()
    prepared.chunks = [(_pdf_chunk(TORQUE_TABLE, 364, "7-9"), 0.3)]
    result = await _ask(
        monkeypatch,
        "Use Table 7-1 in AC_43.13-1B_CHG_1, page 7-9 (PDF page 364) [1].",
        prepared,
    )
    assert result.verification is not None
    assert result.verification.ok, result.verification


async def test_a_page_from_an_uncited_block_is_still_flagged(monkeypatch, mode):
    mode("flag")
    prepared = _prepared()
    prepared.chunks = [
        (_pdf_chunk(TORQUE_TABLE, 364, "7-9"), 0.3),
        (_pdf_chunk(TITANIUM, 349, "6-43"), 0.4),
    ]
    result = await _ask(monkeypatch, "See PDF page 349 [1].", prepared)
    assert result.verification is not None
    assert result.verification.unsupported == ["349"]


async def test_a_wrong_figure_is_still_caught_next_to_a_page_reference(
    monkeypatch, mode
):
    mode("flag")
    prepared = _prepared()
    prepared.chunks = [(_pdf_chunk(TORQUE_TABLE, 364, "7-9"), 0.3)]
    result = await _ask(
        monkeypatch,
        "Page 7-9 (PDF page 364): torque it to 40 inch-pounds [1].",
        prepared,
    )
    assert result.verification is not None
    assert result.verification.unsupported == ["40 inch-pounds"]


# ------------------------------- live query 2026-10-18: "safety wire diameter"

# Real text of the two blocks: one sentence runs from page 7-19 onto 7-20.
PAGE_7_19 = (
    "b. When using double-twist method of safety wiring, .032-inch minimum "
    "diameter wire should be used on parts that have a hole diameter larger than "
    ".045 inch. Safety wire of .020-inch"
)
PAGE_7_20 = (
    "diameter (double strand) may be used on parts having a nominal hole diameter "
    "between .045 and .062 inch with a spacing between parts of less than 2 inches."
)
TABLE_7_8 = (
    "Cable Size Type of Wrap Diameter of Safety Wire Material (Annealed Condition) "
    "5/32 and greater Single 0.057 min. Stainless steel, Monel."
)


def test_min_in_a_table_is_minimum_not_minutes():
    result = verify_answer("Single wrap: 0.057-inch minimum diameter [1].", [TABLE_7_8])
    assert result.ok, result


def test_min_still_counts_against_another_time_unit():
    result = verify_answer("Let it dwell 10 seconds [1].", ["Dwell time is 10 min."])
    assert result.unit_mismatch == ["10 seconds"]


def test_minutes_match_min():
    assert verify_answer("Dwell 10 minutes [1].", ["Dwell time is 10 min."]).ok


def test_a_figure_from_an_uncited_block_is_located():
    answer = "Between .045 and .062 inch, spacing less than 2 inches: .020-inch [1]."
    result = verify_answer(answer, [PAGE_7_19, PAGE_7_20])
    assert result.unsupported == [".062 inch", "2 inches"]
    assert result.found_in_uncited == {".062 inch": [2], "2 inches": [2]}
    assert not result.ok  # the cited block still does not back them


def test_an_invented_figure_is_not_located_anywhere():
    result = verify_answer("Use .071-inch wire [1].", [PAGE_7_19, PAGE_7_20])
    assert result.unsupported == [".071-inch"]
    assert result.found_in_uncited == {}


def test_without_citations_nothing_is_uncited():
    result = verify_answer("Use .071-inch wire.", [PAGE_7_19, PAGE_7_20])
    assert result.found_in_uncited == {}


async def test_the_document_name_in_prose_matches_the_source_line(monkeypatch, mode):
    # Live run 2026-10-20, user-rivet-size-spacing: the answer ended "as specified
    # in AC 43.13-1B CHG 1" and the check flagged 43.13, because the source line
    # writes the name "AC_43.13-1B_CHG_1" and "_" hid the number.
    mode("flag")
    prepared = _prepared()
    prepared.chunks = [(_pdf_chunk(TORQUE_TABLE, 364, "7-9"), 0.3)]
    result = await _ask(
        monkeypatch,
        "Use Table 7-1 [1], as specified in AC 43.13-1B CHG 1.",
        prepared,
    )
    assert result.verification is not None
    assert result.verification.ok, result.verification
