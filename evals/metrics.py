"""Retrieval quality metrics: pure functions, no I/O, fully unit-testable.

These are the scoreboard for Phase 1. Every retrieval change (hybrid,
reranking, ...) must be judged by running the eval harness against these
metrics — never by eyeballing a few queries.
"""


def recall_at_k(ranked_ids: list[int], expected_id: int, k: int) -> float:
    """1.0 if the expected chunk appears in the top k results, else 0.0."""
    return 1.0 if expected_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[int], expected_id: int) -> float:
    """1/rank of the first correct result (1-based); 0.0 if absent.

    MRR (mean reciprocal rank) over many questions tells you how high the
    right answer sits on average — rank 1 scores 1.0, rank 2 scores 0.5.
    """
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id == expected_id:
            return 1.0 / rank
    return 0.0


def aggregate(per_item: list[dict], k_values: tuple[int, ...]) -> dict:
    """Mean metrics over all eval items.

    per_item rows carry recall@k keys (as floats) and "rr". Returns a flat
    dict of means, e.g. {"recall@1": 0.83, "recall@5": 1.0, "mrr": 0.91}.
    """
    if not per_item:
        return {f"recall@{k}": 0.0 for k in k_values} | {"mrr": 0.0}

    summary: dict = {}
    for k in k_values:
        key = f"recall@{k}"
        summary[key] = sum(row[key] for row in per_item) / len(per_item)
    summary["mrr"] = sum(row["rr"] for row in per_item) / len(per_item)
    return summary
