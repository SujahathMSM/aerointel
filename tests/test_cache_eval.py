"""The answer-cache eval: its pairs, its bookkeeping, and the whole loop.

The real numbers need EmbeddingGemma (python -m evals.cache_eval on a developer
machine). CI checks the pair file, the guard-word expectations written in it,
the threshold arithmetic, and runs the loop once against fakes so the script
cannot rot.
"""

import json

import pytest

from app.core.answer_cache import question_guard
from app.services.generation import PreparedAnswer
from evals import cache_eval
from evals.cache_eval import (
    context_difference,
    load_pairs,
    safe_threshold,
    summarise,
    sweep,
    would_hit,
)

PAIRS = load_pairs()


def test_pair_ids_are_unique_and_both_labels_are_present():
    ids = [p["id"] for p in PAIRS]
    assert len(ids) == len(set(ids))
    assert {p["label"] for p in PAIRS} == {"same", "different"}


@pytest.mark.parametrize(
    "pair",
    [p for p in PAIRS if "guard should" in p.get("note", "")],
    ids=lambda p: p["id"],
)
def test_pairs_the_guard_should_separate_are_separated(pair):
    assert question_guard(pair["a"]) != question_guard(pair["b"])


@pytest.mark.parametrize(
    "pair",
    [p for p in PAIRS if "only the threshold" in p.get("note", "")],
    ids=lambda p: p["id"],
)
def test_pairs_left_to_the_threshold_pass_the_guard(pair):
    # If the guard catches one of these, its note (and the eval's point) is stale.
    assert question_guard(pair["a"]) == question_guard(pair["b"])


def _row(
    label, similarity, guard=True, context=True, refused=False, exact=False, kind=None
):
    return {
        "id": f"{label}-{similarity}",
        "label": label,
        "similarity": similarity,
        "guard_equal": guard,
        "same_context": context,
        "context": kind or ("same" if context else "disjoint"),
        "refused": refused,
        "exact": exact,
    }


def test_context_difference():
    assert context_difference([1, 2, 3], [1, 2, 3]) == "same"
    assert context_difference([1, 2, 3], [3, 1, 2]) == "order"
    assert context_difference([1, 2, 3], [1, 2, 4]) == "partial"
    assert context_difference([1, 2], [3, 4]) == "disjoint"


def test_reorder_counts_only_same_passages_in_another_order():
    reordered = _row("same", 0.98, context=False, kind="order")
    partial = _row("same", 0.98, context=False, kind="partial")
    assert not would_hit(reordered, 0.97)  # as built: order must match
    assert would_hit(reordered, 0.97, reorder=True)
    assert not would_hit(partial, 0.97, reorder=True)

    rows = [reordered, partial, _row("different", 0.95, context=False, kind="order")]
    (built,) = sweep(rows, [0.94])
    (reordered_sweep,) = sweep(rows, [0.94], reorder=True)
    assert (built["right_reuses"], built["wrong_reuses"]) == (0, 0)
    assert (reordered_sweep["right_reuses"], reordered_sweep["wrong_reuses"]) == (1, 1)
    assert safe_threshold(rows) == pytest.approx(0.01)
    assert safe_threshold(rows, reorder=True) == pytest.approx(0.96)


def test_would_hit_needs_context_guard_and_similarity():
    assert would_hit(_row("same", 0.98), 0.97)
    assert not would_hit(_row("same", 0.96), 0.97)
    assert not would_hit(_row("same", 0.99, guard=False), 0.97)
    assert not would_hit(_row("same", 0.99, context=False), 0.97)
    assert not would_hit(_row("same", 0.99, refused=True), 0.97)
    assert would_hit(_row("same", 0.5, exact=True), 0.97)  # a repeat always hits
    assert would_hit(_row("same", 0.98, context=None), 0.97)  # unchecked = worst case


def test_sweep_counts_right_and_wrong_reuses():
    rows = [
        _row("same", 0.99),
        _row("same", 0.95),
        _row("same", 0.99, refused=True),  # never reaches the cache: not counted
        _row("different", 0.96),
        _row("different", 0.99, guard=False),
    ]
    at_94, at_97 = sweep(rows, [0.94, 0.97])
    assert (at_94["right_reuses"], at_94["same_total"]) == (2, 2)
    assert (at_94["wrong_reuses"], at_94["different_total"]) == (1, 2)
    assert at_94["wrong_ids"] == ["different-0.96"]
    assert (at_97["right_reuses"], at_97["wrong_reuses"]) == (1, 0)


def test_safe_threshold_is_just_above_the_closest_exposed_different_pair():
    rows = [
        _row("different", 0.93),
        _row("different", 0.95),
        _row("different", 0.999, guard=False),  # the guard handles it
        _row("different", 0.999, context=False),  # never shares a bucket
        _row("same", 0.999),
    ]
    assert safe_threshold(rows) == pytest.approx(0.96)
    assert safe_threshold([_row("same", 0.9)]) == pytest.approx(0.01)


def test_no_threshold_is_safe_for_an_exact_different_pair():
    assert safe_threshold([_row("different", 0.5, exact=True)]) is None
    assert safe_threshold([_row("different", 1.0)]) is None


def test_summary_includes_the_configured_threshold():
    s = summarise([_row("same", 0.99), _row("different", 0.9)], configured=0.975)
    assert 0.975 in [row["threshold"] for row in s["sweep"]]
    assert s["safe_threshold"] == pytest.approx(0.91)
    assert len(s["sweep_if_reordered"]) == len(s["sweep"])
    assert s["context_of_same"] == {"same": 1, "order": 0, "partial": 0, "disjoint": 0}


class _Chunk:
    def __init__(self, cid, content):
        self.id = cid
        self.content = content
        self.metadata_ = {"source": "manual"}


async def test_the_whole_loop_runs_against_fakes(monkeypatch, tmp_path, capsys):
    pairs = [
        {"id": "para", "label": "same", "a": "What is rust?", "b": "Define rust."},
        {
            "id": "size",
            "label": "different",
            "a": "Torque for a 10-32 bolt?",
            "b": "Torque for a 1/4-28 bolt?",
        },
        {"id": "far", "label": "different", "a": "What is rust?", "b": "Cake?"},
    ]
    vectors = {
        "What is rust?": [1.0, 0.0],
        "Define rust.": [1.0, 0.05],
        "Torque for a 10-32 bolt?": [0.0, 1.0],
        "Torque for a 1/4-28 bolt?": [0.0, 1.0],
        "Cake?": [0.7, 0.7],
    }

    class FakeLlm:
        model_name = "fake-model"

        async def embed(self, texts):
            return [vectors[t] for t in texts]

        async def close(self):
            pass

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def fake_prepare(session, llm, question, top_k=None):
        refused = question == "Cake?"
        chunk = _Chunk(2, "torque table") if "bolt" in question else _Chunk(1, "rust")
        return PreparedAnswer(
            question=question,
            refused=refused,
            chunks=[] if refused else [(chunk, 0.2)],
            system=None if refused else "sys",
            prompt=None if refused else "p",
        )

    monkeypatch.setattr(cache_eval, "load_pairs", lambda: pairs)
    monkeypatch.setattr(cache_eval, "make_llm_client", FakeLlm)
    monkeypatch.setattr(cache_eval, "SessionFactory", FakeSession)
    monkeypatch.setattr(cache_eval, "prepare_answer", fake_prepare)

    out = tmp_path / "cache.json"
    assert await cache_eval.run(True, None, out) == 0
    saved = json.loads(out.read_text())
    rows = {r["id"]: r for r in saved["rows"]}
    assert rows["para"]["same_context"] is True and rows["para"]["guard_equal"]
    assert rows["para"]["context"] == "same"
    assert rows["para"]["chunks_a"] == rows["para"]["chunks_b"] == [1]
    assert rows["size"]["guard_equal"] is False  # 10-32 vs 1/4-28
    assert rows["size"]["similarity"] == pytest.approx(1.0)
    assert rows["far"]["refused"] is True
    configured = next(
        s
        for s in saved["summary"]["sweep"]
        if s["threshold"] == saved["summary"]["configured_threshold"]
    )
    assert configured["right_reuses"] == 1 and configured["wrong_reuses"] == 0
    assert "Lowest threshold with no wrong reuse" in capsys.readouterr().out
