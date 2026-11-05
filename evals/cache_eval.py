"""Eval for the semantic answer cache: which threshold is safe?

The cache reuses an answer when two questions (1) get the identical context,
(2) share their guard words (numbers, negations, min/max-type words) and
(3) have embeddings at least ANSWER_CACHE_MIN_SIMILARITY alike. Only the third
is a free parameter, and only the real embedding model can say where to put it.

`evals/cache_pairs.json` holds labelled pairs: "same" (A's answer is a correct
answer to B) and "different" (it is not). For each pair this embeds both
questions, applies the guard and, unless --no-retrieval, runs both through the
API's own `prepare_answer()` to see whether they would land in the same cache
bucket. Then, for each threshold, it counts:

  wrong reuses   "different" pairs that would get each other's answer. Must be 0.
  right reuses   "same" pairs that would hit (the point of having the cache).

and recommends the lowest threshold with no wrong reuse, plus a margin.

It also says *why* two questions' contexts differ: the same passages in another
order ("order"), some passages shared ("partial") or none ("disjoint"). The
cache needs identical order, because a cached answer's [n] citations point at
positions. The "if reordered" columns show what renumbering those citations
would buy: hits counted as if same-passages-other-order were allowed.

    python -m evals.cache_eval                     # needs Ollama + Postgres with the manual
    python -m evals.cache_eval --no-retrieval      # Ollama only; assumes every pair
                                                   # shares its context (worst case)
    python -m evals.cache_eval --save evals/results/cache.json

Retrieval follows the API's settings (HYBRID_ENABLED, MAX_DISTANCE, ...), and
with ANSWER_CHECK on, `prepare_answer()` asks the model its yes/no question too.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.core.answer_cache import cosine_similarity, normalize_question, question_guard
from app.core.config import get_settings
from app.db.base import SessionFactory
from app.llm.anthropic import make_llm_client
from app.services.generation import cache_bucket, prepare_answer

PAIRS_PATH = Path(__file__).parent / "cache_pairs.json"
# Thresholds the table shows; the configured one is added if it is not here.
GRID = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
# Added above the most similar "different" pair that would otherwise hit, so
# a paraphrase the set does not contain has some room.
MARGIN = 0.01


def load_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    if not pairs:
        raise RuntimeError(f"{path} has no pairs")
    for pair in pairs:
        if pair["label"] not in ("same", "different"):
            raise RuntimeError(f"{pair['id']}: label must be 'same' or 'different'")
    return pairs


def context_difference(ids_a: list[int], ids_b: list[int]) -> str:
    """How two questions' context chunks differ: same, order, partial, disjoint."""
    if ids_a == ids_b:
        return "same"
    if sorted(ids_a) == sorted(ids_b):
        return "order"
    if set(ids_a) & set(ids_b):
        return "partial"
    return "disjoint"


def _shares_context(row: dict, reorder: bool) -> bool:
    if row["same_context"] is not False:  # identical, or not checked (worst case)
        return True
    return reorder and row.get("context") == "order"


def would_hit(row: dict, threshold: float, reorder: bool = False) -> bool:
    """Would the cache hand B the answer written for A, at this threshold?

    reorder: also count the same passages in another order as shared (what
    renumbering the cached answer's citations would allow).
    """
    if row["refused"] or not row["guard_equal"] or not _shares_context(row, reorder):
        return False
    return row["exact"] or row["similarity"] >= threshold


def sweep(
    rows: list[dict], thresholds: list[float], reorder: bool = False
) -> list[dict]:
    """Right and wrong reuses at each threshold (pure, so it is unit-tested)."""
    same = [r for r in rows if r["label"] == "same" and not r["refused"]]
    different = [r for r in rows if r["label"] == "different" and not r["refused"]]
    return [
        {
            "threshold": t,
            "right_reuses": sum(would_hit(r, t, reorder) for r in same),
            "same_total": len(same),
            "wrong_reuses": sum(would_hit(r, t, reorder) for r in different),
            "wrong_ids": [r["id"] for r in different if would_hit(r, t, reorder)],
            "different_total": len(different),
        }
        for t in thresholds
    ]


def safe_threshold(rows: list[dict], reorder: bool = False) -> float | None:
    """Lowest threshold (plus MARGIN) at which no "different" pair would hit.

    None when even 1.0 is not safe: a "different" pair that matches exactly or
    has identical embeddings cannot be told apart by any threshold.
    """
    exposed = [
        r
        for r in rows
        if r["label"] == "different"
        and not r["refused"]
        and _shares_context(r, reorder)
        and r["guard_equal"]
    ]
    if any(r["exact"] or r["similarity"] >= 1.0 for r in exposed):
        return None
    highest = max((r["similarity"] for r in exposed), default=0.0)
    return min(1.0, round(highest + MARGIN, 3))


def _context_counts(rows: list[dict]) -> dict[str, int]:
    kinds = ["same", "order", "partial", "disjoint"]
    return {k: sum(r.get("context") == k for r in rows) for k in kinds}


def summarise(rows: list[dict], configured: float) -> dict:
    thresholds = sorted({*GRID, configured})
    same = [r for r in rows if r["label"] == "same" and not r["refused"]]
    different = [r for r in rows if r["label"] == "different" and not r["refused"]]
    return {
        "pairs": len(rows),
        "refused": sum(r["refused"] for r in rows),
        "different_caught_by_guard": sum(not r["guard_equal"] for r in different),
        "different_other_context": sum(r["same_context"] is False for r in different),
        "context_of_same": _context_counts(same),
        "context_of_different": _context_counts(different),
        "safe_threshold": safe_threshold(rows),
        "safe_threshold_if_reordered": safe_threshold(rows, reorder=True),
        "configured_threshold": configured,
        "sweep": sweep(rows, thresholds),
        "sweep_if_reordered": sweep(rows, thresholds, reorder=True),
    }


def _why(row: dict) -> str:
    if row["refused"]:
        return "refused"
    if row["same_context"] is False:
        return {"order": "passage order"}.get(
            row.get("context") or "", "other passages"
        )
    if not row["guard_equal"]:
        return "guard"
    if row["exact"]:
        return "exact"
    return ""


def _cell(s: dict, key: str, total: str) -> str:
    return f"{s[key]}/{s[total]}"


def _safe_line(safe: float | None, what: str) -> str:
    if safe is None:
        return f"{what}: none is safe (a 'different' pair matches exactly; add a guard word)"
    return f"{what}: {safe}"


def print_report(rows: list[dict], summary: dict) -> None:
    print()
    print(f"{'pair':<38} {'label':>9} {'similarity':>10}  blocked by")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: (r["label"], -r["similarity"])):
        print(f"{r['id']:<38} {r['label']:>9} {r['similarity']:>10.4f}  {_why(r)}")
    print("-" * 78)
    print(f"{'':>9} {'as built':>27}   {'if reordered':>27}")
    print(f"{'threshold':>9} {'right':>13} {'wrong':>13}   {'right':>13} {'wrong':>13}")
    for s, r in zip(summary["sweep"], summary["sweep_if_reordered"], strict=True):
        mark = (
            "  <- configured"
            if s["threshold"] == summary["configured_threshold"]
            else ""
        )
        print(
            f"{s['threshold']:>9.2f} "
            f"{_cell(s, 'right_reuses', 'same_total'):>13} "
            f"{_cell(s, 'wrong_reuses', 'different_total'):>13}   "
            f"{_cell(r, 'right_reuses', 'same_total'):>13} "
            f"{_cell(r, 'wrong_reuses', 'different_total'):>13}{mark}"
        )
        wrong = sorted({*s["wrong_ids"], *r["wrong_ids"]})
        if wrong:
            print(f"{'':>10}wrong: {', '.join(wrong)}")
    print("-" * 78)
    for label in ("same", "different"):
        c = summary[f"context_of_{label}"]
        print(
            f"'{label}' pairs' passages: identical {c['same']}, other order "
            f"{c['order']}, partly shared {c['partial']}, none shared {c['disjoint']}"
        )
    print(
        f"'different' pairs blocked by the guard words: "
        f"{summary['different_caught_by_guard']}; by a different context: "
        f"{summary['different_other_context']}; refused (never cached): "
        f"{summary['refused']} pair(s)"
    )
    print(
        _safe_line(
            summary["safe_threshold"],
            f"Lowest threshold with no wrong reuse, plus {MARGIN}",
        )
        + f"  (configured: {summary['configured_threshold']})"
    )
    print(_safe_line(summary["safe_threshold_if_reordered"], "  same, if reordered"))


async def run(retrieval: bool, only: set[str] | None, save: Path | None) -> int:
    pairs = [p for p in load_pairs() if only is None or p["id"] in only]
    settings = get_settings()
    llm = make_llm_client()
    rows: list[dict] = []
    started = time.monotonic()

    async def context(question: str) -> tuple[str, list[int]] | None:
        """The API's cache bucket and context chunk ids; None if it would refuse."""
        async with SessionFactory() as session:
            prepared = await prepare_answer(session, llm, question)
        if prepared.refused:
            return None
        return cache_bucket(prepared, llm), [c.id for c, _ in prepared.chunks]

    try:
        for pair in pairs:
            embedding_a, embedding_b = await llm.embed([pair["a"], pair["b"]])
            row = {
                "id": pair["id"],
                "label": pair["label"],
                "note": pair.get("note", ""),
                "a": pair["a"],
                "b": pair["b"],
                "similarity": round(cosine_similarity(embedding_a, embedding_b), 6),
                "exact": normalize_question(pair["a"]) == normalize_question(pair["b"]),
                "guard_a": list(question_guard(pair["a"])),
                "guard_b": list(question_guard(pair["b"])),
                "refused": False,
                "same_context": None,  # None = not checked (--no-retrieval)
                "context": None,  # same | order | partial | disjoint
                "chunks_a": None,
                "chunks_b": None,
            }
            row["guard_equal"] = row["guard_a"] == row["guard_b"]
            if retrieval:
                got_a, got_b = await context(pair["a"]), await context(pair["b"])
                # A refused question never reaches the cache, either way round.
                row["refused"] = got_a is None or got_b is None
                if got_a is not None and got_b is not None:
                    row["same_context"] = got_a[0] == got_b[0]
                    row["chunks_a"], row["chunks_b"] = got_a[1], got_b[1]
                    row["context"] = context_difference(got_a[1], got_b[1])
            rows.append(row)
            print(f"  {pair['id']}", flush=True)
    finally:
        await llm.close()

    summary = summarise(rows, settings.answer_cache_min_similarity)
    label = (
        f"embedding_model={settings.embedding_model}  "
        f"retrieval={'on' if retrieval else 'off (every pair assumed same context)'}  "
        f"hybrid={settings.hybrid_enabled}  max_distance={settings.max_distance}  "
        f"top_k={settings.top_k}  model={llm.model_name}  "
        f"{time.monotonic() - started:.0f}s"
    )
    print()
    print(label)
    print_report(rows, summary)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps({"label": label, "summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nsaved {save}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval for the semantic answer cache.")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="skip Postgres: treat every pair as sharing its context (worst case)",
    )
    parser.add_argument(
        "--only", action="append", metavar="ID", help="run only this pair id"
    )
    parser.add_argument("--save", type=Path, help="write all rows here")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            run(
                not args.no_retrieval,
                set(args.only) if args.only else None,
                args.save,
            )
        )
    )
