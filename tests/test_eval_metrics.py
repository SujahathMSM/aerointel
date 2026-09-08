import json
from pathlib import Path

from evals.metrics import aggregate, recall_at_k, reciprocal_rank

GOLDEN_SET = Path(__file__).parent.parent / "evals" / "golden_set.json"


# --- recall@k -----------------------------------------------------------


def test_recall_at_k_hit_within_k():
    assert recall_at_k([10, 20, 30], 20, k=3) == 1.0


def test_recall_at_k_miss_beyond_k():
    assert recall_at_k([10, 20, 30, 40], 40, k=3) == 0.0


def test_recall_at_k_absent():
    assert recall_at_k([10, 20, 30], 99, k=5) == 0.0


def test_recall_at_k_rank_one():
    assert recall_at_k([7, 8, 9], 7, k=1) == 1.0


# --- reciprocal rank ----------------------------------------------------


def test_rr_rank_one_scores_one():
    assert reciprocal_rank([5, 6, 7], 5) == 1.0


def test_rr_rank_two_scores_half():
    assert reciprocal_rank([5, 6, 7], 6) == 0.5


def test_rr_absent_scores_zero():
    assert reciprocal_rank([5, 6, 7], 99) == 0.0


# --- aggregate ----------------------------------------------------------


def test_aggregate_means():
    per_item = [
        {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 1.0},
        {"recall@1": 0.0, "recall@3": 1.0, "recall@5": 1.0, "rr": 0.5},
    ]
    summary = aggregate(per_item, (1, 3, 5))
    assert summary["recall@1"] == 0.5
    assert summary["recall@3"] == 1.0
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == 0.75


def test_aggregate_empty():
    summary = aggregate([], (1, 3, 5))
    assert summary == {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0}


# --- golden set sanity ---------------------------------------------------


def test_golden_set_is_wellformed():
    items = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    assert len(items) >= 10, "golden set should have a meaningful number of items"
    for item in items:
        assert item["question"].strip()
        assert item["expected"]["source"].strip()
        assert item["expected"]["section"].strip()
