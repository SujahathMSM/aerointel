"""The reading limit (CONTEXT_MAX_DISTANCE) and the answer check (ANSWER_CHECK).

Two misses on the FAA manual had one cause: the question passed the 0.48 gate
on one chunk, but the chunk holding the answer ranked next at 0.486 / 0.495 and
the same 0.48 limit kept it from the model. The reading limit separates "answer
at all?" (the gate, unchanged) from "read what?".

The answer check asks the model, before it answers, whether the passages it
would read answer the question: the distance gate measures topic, and a
near-topic question ("torque for a helicopter rotor bolt") sits as close to
the torque table as a real torque question does.
"""

import json

import pytest

from app.api.v1 import router as router_module
from app.core.config import Settings, get_settings
from app.services import generation
from app.services.generation import (
    CHECK_NO,
    CHECK_UNCLEAR,
    CHECK_YES,
    REASON_DISTANCE,
    REASON_NOT_IN_CONTEXT,
    answer_check_prompt,
    context_ceiling,
    parse_answer_check,
    refusal_message,
    select_context,
)


class Chunk:
    def __init__(self, cid, content="text", page=1):
        self.id = cid
        self.content = content
        self.page = page
        self.page_label = None
        self.document_id = 1
        self.chunk_index = cid
        self.metadata_ = {"source": "manual"}


FLUX = Chunk(1, "Remove all flux after welding the tank.", page=463)
PRECAUTIONS = Chunk(
    2, "e. Metal fuel tanks must not be welded unless adequately purged.", page=464
)
FAR = Chunk(3, "Cadmium plating finishes.", page=369)

# the weld question as served live: passes the gate on FLUX, answer at 0.495
WELD = [(FLUX, 0.422), (PRECAUTIONS, 0.495), (FAR, 0.585)]


# ---------------------------------------------------------- select_context


def test_without_a_reading_limit_the_answer_just_past_the_gate_is_dropped():
    assert [c.id for c, _ in select_context(WELD, 0.48)] == [1]


def test_the_reading_limit_lets_it_through_in_retrieval_order():
    read = select_context(WELD, 0.48, 0.52)
    assert [(c.id, d) for c, d in read] == [(1, 0.422), (2, 0.495)]


def test_the_reading_limit_never_turns_a_refusal_into_an_answer():
    closest_is_past_the_gate = [(PRECAUTIONS, 0.495), (FAR, 0.51)]
    assert select_context(closest_is_past_the_gate, 0.48, 0.52) == []
    assert select_context([], 0.48, 0.52) == []


def test_a_reading_limit_below_the_gate_cannot_tighten_it():
    assert context_ceiling(0.48, 0.30) == 0.48
    assert context_ceiling(0.48, None) == 0.48
    assert context_ceiling(0.48, 0.52) == 0.52
    assert [c.id for c, _ in select_context(WELD, 0.48, 0.30)] == [1]


def test_with_hybrid_order_the_gate_looks_at_every_chunk_not_the_first():
    fused_order = [(FAR, 0.55), (FLUX, 0.42)]  # keyword rank put FAR first
    assert [c.id for c, _ in select_context(fused_order, 0.48)] == [1]


def test_a_blank_setting_means_unset():
    assert Settings(context_max_distance="").context_max_distance is None
    assert Settings(context_max_distance="0.52").context_max_distance == 0.52
    assert Settings().answer_check == "off"


# --------------------------------------------------------- answer check reply


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("YES", CHECK_YES),
        ("Yes.", CHECK_YES),
        ("yes, block [2] gives it", CHECK_YES),
        ("NO", CHECK_NO),
        ("No - the context covers fixed-wing aircraft only.", CHECK_NO),
        ("**NO**", CHECK_NO),
        ("", CHECK_UNCLEAR),
        ("The context mentions torque values.", CHECK_UNCLEAR),
        ("Nope", CHECK_UNCLEAR),
        ("Yesterday", CHECK_UNCLEAR),
    ],
)
def test_the_reply_is_read_from_its_first_word_only(reply, expected):
    assert parse_answer_check(reply) == expected


def test_the_check_reads_the_same_numbered_passages_the_answer_would():
    prompt = answer_check_prompt("Can a fuel tank be welded?", WELD[:2])
    assert "[1] Source: manual" in prompt and "[2] Source: manual" in prompt
    assert "adequately purged" in prompt
    assert "QUESTION: Can a fuel tank be welded?" in prompt
    assert prompt.endswith("REPLY (YES or NO):")
    assert "ANSWER:" not in prompt


def test_the_refusal_says_why():
    assert "do not answer the question" in refusal_message(
        REASON_NOT_IN_CONTEXT, [], [], 0.48
    )


# -------------------------------------------------------- prepare_answer wiring


class FakeSession:
    async def commit(self):
        pass


class FakeOllama:
    """Records every chat call; answers the check with `verdict`."""

    def __init__(self, verdict="YES"):
        self.verdict = verdict
        self.calls: list[str] = []

    async def chat(self, system, user):
        if user.endswith("REPLY (YES or NO):"):
            self.calls.append("check")
            return self.verdict
        self.calls.append("answer")
        return "Metal tanks must be purged first [2]."


def _use(monkeypatch, retrieved, **settings):
    values = get_settings().model_copy(update=settings)
    monkeypatch.setattr(generation, "get_settings", lambda: values)

    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        return retrieved

    monkeypatch.setattr(generation, "retrieve", fake_retrieve)


async def test_prepare_answer_reads_past_the_gate_when_configured(monkeypatch):
    _use(monkeypatch, WELD, context_max_distance=0.52)
    prepared = await generation.prepare_answer(FakeSession(), FakeOllama(), "q?")
    assert prepared.refused is False
    assert [c.id for c, _ in prepared.chunks] == [1, 2]
    assert "adequately purged" in prepared.prompt


async def test_prepare_answer_still_refuses_when_the_closest_chunk_is_too_far(
    monkeypatch,
):
    _use(monkeypatch, [(PRECAUTIONS, 0.495)], context_max_distance=0.60)
    ollama = FakeOllama()
    prepared = await generation.prepare_answer(FakeSession(), ollama, "q?")
    assert prepared.refused is True
    assert prepared.refusal_reason == REASON_DISTANCE
    assert ollama.calls == []


async def test_check_off_makes_no_extra_call(monkeypatch):
    _use(monkeypatch, WELD, answer_check="off")
    ollama = FakeOllama()
    result = await generation.answer_question(FakeSession(), ollama, "q?")
    assert ollama.calls == ["answer"]
    assert result.answer_check is None


async def test_check_flag_reports_but_still_answers(monkeypatch):
    _use(monkeypatch, WELD, answer_check="flag")
    ollama = FakeOllama(verdict="NO")
    result = await generation.answer_question(FakeSession(), ollama, "q?")
    assert ollama.calls == ["check", "answer"]
    assert result.refused is False
    assert result.answer_check == CHECK_NO


async def test_check_refuse_stops_before_the_answer_on_no(monkeypatch):
    _use(monkeypatch, WELD, answer_check="refuse")
    ollama = FakeOllama(verdict="NO")
    result = await generation.answer_question(FakeSession(), ollama, "q?")
    assert ollama.calls == ["check"]  # the answer was never generated
    assert result.refused is True
    assert result.refusal_reason == REASON_NOT_IN_CONTEXT
    assert result.answer_check == CHECK_NO
    assert [c.id for c, _ in result.chunks] == [1]  # the passages it judged


@pytest.mark.parametrize("verdict", ["YES", "I think the table covers it"])
async def test_check_refuse_answers_on_yes_and_on_an_unclear_reply(
    monkeypatch, verdict
):
    _use(monkeypatch, WELD, answer_check="refuse")
    ollama = FakeOllama(verdict=verdict)
    result = await generation.answer_question(FakeSession(), ollama, "q?")
    assert ollama.calls == ["check", "answer"]
    assert result.refused is False


async def test_the_check_never_runs_for_a_question_the_gate_refused(monkeypatch):
    _use(monkeypatch, [(FAR, 0.9)], answer_check="refuse")
    ollama = FakeOllama()
    result = await generation.answer_question(FakeSession(), ollama, "q?")
    assert result.refused is True
    assert result.refusal_reason == REASON_DISTANCE
    assert ollama.calls == []


async def test_the_check_sees_the_widened_reading_set(monkeypatch):
    _use(monkeypatch, WELD, answer_check="flag", context_max_distance=0.52)
    seen: list[str] = []

    class Spy(FakeOllama):
        async def chat(self, system, user):
            seen.append(user)
            return await super().chat(system, user)

    await generation.answer_question(FakeSession(), Spy(), "q?")
    assert "adequately purged" in seen[0]  # the 0.495 passage was judged too


# ------------------------------------------------------------------ the API


def _prepared(**overrides) -> generation.PreparedAnswer:
    values = {
        "question": "q?",
        "refused": False,
        "chunks": [(FLUX, 0.422)],
        "system": "sys",
        "prompt": "prompt",
        "prompt_version": "v1",
        "answer_check": CHECK_YES,
    }
    values.update(overrides)
    return generation.PreparedAnswer(**values)


@pytest.mark.redis
async def test_query_reports_the_check_and_a_not_in_context_refusal(
    client, monkeypatch
):
    async def fake_answer(session, ollama, question, top_k=None):
        return generation.AnswerResult(
            question=question,
            refused=True,
            chunks=[(FLUX, 0.422)],
            refusal_reason=REASON_NOT_IN_CONTEXT,
            answer_check=CHECK_NO,
        )

    monkeypatch.setattr(router_module, "answer_question", fake_answer)
    body = (await client.post("/v1/query", json={"question": "q?"})).json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "not_in_context"
    assert body["answer_check"] == "no"
    assert "do not answer the question" in body["refusal_message"]


@pytest.mark.redis
async def test_stream_meta_carries_the_check_and_never_streams_a_refusal(
    client, monkeypatch
):
    calls: list[str] = []

    async def fake_prepare(session, ollama, question, top_k=None):
        return _prepared(
            refused=True,
            system=None,
            prompt=None,
            refusal_reason=REASON_NOT_IN_CONTEXT,
            answer_check=CHECK_NO,
        )

    def fake_stream(ollama, prepared):
        calls.append("stream")
        raise AssertionError("a refused question was streamed")

    monkeypatch.setattr(router_module, "prepare_answer", fake_prepare)
    monkeypatch.setattr(router_module, "stream_answer", fake_stream)
    response = await client.post("/v1/query/stream", json={"question": "q?"})
    events = [
        (block.split("\n")[0].removeprefix("event: "), block.split("\n")[1])
        for block in response.text.strip().split("\n\n")
    ]
    assert [name for name, _ in events] == ["meta", "done"]
    meta = json.loads(events[0][1].removeprefix("data: "))
    assert meta["answer_check"] == "no"
    assert meta["refusal_reason"] == "not_in_context"
    assert calls == []


# ------------------------------------------------- answer-check prompt version


def test_the_check_prompt_is_pinned_to_the_measured_version():
    # v2: 24/24 YES on right questions and 13/13 NO on gate misses, 3 runs
    # (evals/results/check_v2*.json); v1 said NO to fabric-punch-tester.
    assert Settings().answer_check_version == "v2"
    assert generation.answer_check_system().version == "v2"


def test_a_blank_version_means_the_newest_prompt(monkeypatch):
    monkeypatch.setenv("ANSWER_CHECK_VERSION", "")
    from app.prompts import latest_version

    values = Settings()
    assert values.answer_check_version is None
    monkeypatch.setattr(generation, "get_settings", lambda: values)
    assert generation.answer_check_system().version == latest_version("answer_check")


async def test_an_explicit_version_reaches_the_model(monkeypatch):
    seen = []

    class Recorder:
        async def chat(self, system, prompt):
            seen.append(system)
            return "YES"

    v2 = generation.get_prompt("answer_check", "v2")
    verdict = await generation.check_answerable(Recorder(), "q", WELD[:1], "v2")
    assert verdict == CHECK_YES
    assert seen == [v2.text]
    assert "how reliable" in v2.text  # judgement questions are answerable
