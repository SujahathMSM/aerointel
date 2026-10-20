"""The live trust-layer eval: its reading of answers and its bookkeeping.

The real numbers need Ollama (python -m evals.live_eval on a developer
machine). CI checks that the decline heuristic and the counts are right, and
runs the whole loop once against fakes so the script cannot rot.
"""

import pytest

from evals import live_eval
from evals.live_eval import looks_like_decline, summarise_live


@pytest.mark.parametrize(
    "answer",
    [
        "The context does not specify a torque for helicopter rotor blade bolts.",
        "The CONTEXT doesn't mention the Boeing 737.",
        "This is not covered in the provided passages.",
        "No information about carbon fiber propellers is given [1].",
        "The wire diameter for the A320 cannot be determined from the context.",
        "I am unable to answer from the context.",
    ],
)
def test_declines_are_recognised(answer):
    assert looks_like_decline(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Use .032-inch safety wire; larger than .045 inch is not used [1].",
        "Metal fuel tanks must not be welded or soldered unless purged [2].",
        "Do not reuse cotter pins [1].",
    ],
)
def test_ordinary_answers_are_not_declines(answer):
    assert not looks_like_decline(answer)


def _row(kind, gate=True, check="yes", answer="text", declined=False, ok=True, n=1):
    return {
        "id": "x",
        "kind": kind,
        "gate_passed": gate,
        "check": check if gate else None,
        "answer": answer if gate else None,
        "declined": declined,
        "verification_ok": ok,
        "numbers_checked": n,
        "facts_total": 2,
        "facts_in_answer": 2 if ok else 1,
    }


def test_summary_counts():
    right = [
        _row("right"),
        _row("right", check="no", declined=True),
        _row("right", ok=False),
        _row("right", gate=False),
    ]
    off = [
        _row("off_topic", check="no", declined=True),
        _row("off_topic"),
        _row("off_topic", gate=False),
    ]
    s = summarise_live(right, off)
    assert (s["right_total"], s["right_passed_gate"]) == (4, 3)
    assert s["right_check_no"] == 1  # would be wrongly refused
    assert s["right_declined"] == 1
    assert s["right_flagged"] == 1
    assert s["right_all_facts_in_answer"] == 2
    assert (s["off_total"], s["off_passed_gate"]) == (3, 2)
    assert s["off_check_no"] == 1  # caught
    assert s["off_declined"] == 1


def test_off_topic_is_counted_per_group():
    off = [
        {**_row("off_topic", check="no", declined=True), "group": "near_topic"},
        {**_row("off_topic"), "group": "near_topic"},
        {**_row("off_topic", gate=False), "group": "outside"},
        _row("off_topic", gate=False),  # no group: only in the totals
    ]
    s = summarise_live([], off)
    assert s["off_by_group"] == {
        "near_topic": {
            "total": 2,
            "passed_gate": 2,
            "check_no": 1,
            "answered": 2,
            "declined": 1,
        },
        "outside": {
            "total": 1,
            "passed_gate": 0,
            "check_no": 0,
            "answered": 0,
            "declined": 0,
        },
    }
    assert s["off_total"] == 4


def test_check_only_rows_have_no_answers():
    rows = [_row("right", answer=None)]
    s = summarise_live(rows, [])
    assert s["right_answered"] == 0 and s["right_passed_gate"] == 1


class Chunk:
    def __init__(self, cid, page, content):
        self.id = cid
        self.page = page
        self.page_label = None
        self.content = content
        self.metadata_ = {"source": "manual"}


async def test_the_whole_loop_runs_against_fakes(monkeypatch, tmp_path, capsys):
    golden = {
        "source": "manual",
        "items": [
            {
                "id": "weld",
                "question": "Can a fuel tank be welded?",
                "pages": [464],
                "facts": ["adequately purged"],
            }
        ],
        "off_topic": [{"id": "cake", "question": "How do I bake a cake?"}],
    }
    answer_chunk = Chunk(2, 464, "Tanks must not be welded unless adequately purged.")

    async def fake_retrieve(session, ollama, question, top_k=None, **kw):
        if "cake" in question:
            return [(Chunk(9, 1, "Rivets."), 0.80)]
        return [(Chunk(1, 463, "Remove flux."), 0.42), (answer_chunk, 0.495)]

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            pass

    class FakeOllama:
        model_name = "fake-model"

        async def chat(self, system, user):
            if user.endswith("REPLY (YES or NO):"):
                return "YES"
            return "They must be adequately purged first [2]."

        async def close(self):
            pass

    monkeypatch.setattr(live_eval, "load_golden", lambda: golden)
    monkeypatch.setattr(live_eval, "retrieve", fake_retrieve)
    monkeypatch.setattr(live_eval, "SessionFactory", FakeSession)
    monkeypatch.setattr(live_eval, "make_llm_client", FakeOllama)

    out = tmp_path / "live.json"
    assert await live_eval.run("hybrid", 0.52, False, None, out) == 0
    import json

    saved = json.loads(out.read_text())
    weld, cake = saved["right"][0], saved["off_topic"][0]
    assert weld["gate_passed"] and weld["read_right_page"]  # read past the gate
    assert weld["check"] == "yes" and weld["facts_in_answer"] == 1
    assert weld["declined"] is False
    assert cake["gate_passed"] is False and cake["answer"] is None
    assert "1/1 pass the gate" in capsys.readouterr().out
