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


def would_hit(row: dict, threshold: float) -> bool:
    """Would the cache hand B the answer written for A, at this threshold?"""
    if row["refused"] or row["same_context"] is False or not row["guard_equal"]:
        return False
    return row["exact"] or row["similarity"] >= threshold


def sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """Right and wrong reuses at each threshold (pure, so it is unit-tested)."""
    same = [r for r in rows if r["label"] == "same" and not r["refused"]]
    different = [r for r in rows if r["label"] == "different" and not r["refused"]]
    return [
        {
            "threshold": t,
            "right_reuses": sum(would_hit(r, t) for r in same),
            "same_total": len(same),
            "wrong_reuses": sum(would_hit(r, t) for r in different),
            "wrong_ids": [r["id"] for r in different if would_hit(r, t)],
            "different_total": len(different),
        }
        for t in thresholds
    ]


def safe_threshold(rows: list[dict]) -> float | None:
    """Lowest threshold (plus MARGIN) at which no "different" pair would hit.

    None when even 1.0 is not safe: a "different" pair that matches exactly or
    has identical embeddings cannot be told apart by any threshold.
    """
    exposed = [
        r
        for r in rows
        if r["label"] == "different"
        and not r["refused"]
        and r["same_context"] is not False
        and r["guard_equal"]
    ]
    if any(r["exact"] or r["similarity"] >= 1.0 for r in exposed):
        return None
    highest = max((r["similarity"] for r in exposed), default=0.0)
    return min(1.0, round(highest + MARGIN, 3))


def summarise(rows: list[dict], configured: float) -> dict:
    thresholds = sorted({*GRID, configured})
    different = [r for r in rows if r["label"] == "different" and not r["refused"]]
    return {
        "pairs": len(rows),
        "refused": sum(r["refused"] for r in rows),
        "different_caught_by_guard": sum(not r["guard_equal"] for r in different),
        "different_other_context": sum(r["same_context"] is False for r in different),
        "safe_threshold": safe_threshold(rows),
        "configured_threshold": configured,
        "sweep": sweep(rows, thresholds),
    }


def _why(row: dict) -> str:
    if row["refused"]:
        return "refused"
    if row["same_context"] is False:
        return "other context"
    if not row["guard_equal"]:
        return "guard"
    if row["exact"]:
        return "exact"
    return ""


def print_report(rows: list[dict], summary: dict) -> None:
    print()
    print(f"{'pair':<38} {'label':>9} {'similarity':>10}  blocked by")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: (r["label"], -r["similarity"])):
        print(f"{r['id']:<38} {r['label']:>9} {r['similarity']:>10.4f}  {_why(r)}")
    print("-" * 74)
    print(f"{'threshold':>9} {'right reuses':>14} {'wrong reuses':>14}")
    for s in summary["sweep"]:
        mark = (
            "  <- configured"
            if s["threshold"] == summary["configured_threshold"]
            else ""
        )
        wrong = f"{s['wrong_reuses']}/{s['different_total']}"
        print(
            f"{s['threshold']:>9.2f} {s['right_reuses']:>9}/{s['same_total']:<4} "
            f"{wrong:>14}{mark}"
        )
        if s["wrong_ids"]:
            print(f"{'':>10}wrong: {', '.join(s['wrong_ids'])}")
    print("-" * 74)
    print(
        f"'different' pairs blocked by the guard words: "
        f"{summary['different_caught_by_guard']}; by a different context: "
        f"{summary['different_other_context']}; refused (never cached): "
        f"{summary['refused']} pair(s)"
    )
    safe = summary["safe_threshold"]
    if safe is None:
        print(
            "No threshold is safe: a 'different' pair is identical to the cache. "
            "Add a guard word for it."
        )
    else:
        print(
            f"Lowest threshold with no wrong reuse, plus {MARGIN}: {safe}  "
            f"(configured: {summary['configured_threshold']})"
        )


async def run(retrieval: bool, only: set[str] | None, save: Path | None) -> int:
    pairs = [p for p in load_pairs() if only is None or p["id"] in only]
    settings = get_settings()
    llm = make_llm_client()
    rows: list[dict] = []
    started = time.monotonic()

    async def bucket(question: str) -> str | None:
        """The cache bucket the API would use; None if it would refuse."""
        async with SessionFactory() as session:
            prepared = await prepare_answer(session, llm, question)
        return None if prepared.refused else cache_bucket(prepared, llm)

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
            }
            row["guard_equal"] = row["guard_a"] == row["guard_b"]
            if retrieval:
                bucket_a, bucket_b = await bucket(pair["a"]), await bucket(pair["b"])
                # A refused question never reaches the cache, either way round.
                row["refused"] = bucket_a is None or bucket_b is None
                row["same_context"] = None if row["refused"] else bucket_a == bucket_b
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
