"""Retrieval eval for real ingested documents (scored by page, not chunk id).

`run_eval` scores the 6 synthetic chunks by chunk id. A real manual is
different: ingesting it again gives every chunk a new id, but page 7-39 stays
page 7-39. So a question here lists the PDF page(s) that answer it, and a
retrieved chunk is a hit when it comes from one of those pages.

    python -m evals.doc_eval                          # vector search baseline
    python -m evals.doc_eval --retriever hybrid       # + keyword search (RRF)
    python -m evals.doc_eval --neighbors 1            # + the chunk after each hit
    python -m evals.doc_eval --save evals/results/baseline.json

Needs Postgres with the document ingested (scripts/ingest_pdfs.py) and Ollama
running for the embedding model. It reports four things:

- recall@k / MRR: does the right page show up, and how high?
- reaches prompt: does a right-page chunk pass the MAX_DISTANCE gate, i.e. is
  the question answered rather than refused?
- context hit and answer facts: is a right page among the chunks the model would
  actually read (after widening, with --neighbors), and do the key phrases of the
  answer appear in them? Retrieving the right page can still leave out the
  sentence that answers the question; the facts check sees that.
- refusals: off-topic questions must be refused. The distances of both groups
  show where MAX_DISTANCE should sit.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.base import SessionFactory
from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient
from app.services.context import expand_with_neighbours
from app.services.generation import CHARS_PER_TOKEN
from app.services.pipeline import retrieve
from app.services.reranking import build_reranker
from evals.metrics import hit_recall_at_k, hit_reciprocal_rank

GOLDEN_PATH = Path(__file__).parent / "golden_faa.json"
RETRIEVE_K = 10
SCORED_K_VALUES = (1, 3, 5)


def load_golden(path: Path = GOLDEN_PATH) -> dict:
    golden = json.loads(path.read_text(encoding="utf-8"))
    if not golden.get("items"):
        raise RuntimeError(f"{path} has no questions")
    return golden


def is_hit(chunk, source: str, pages: list[int]) -> bool:
    """A chunk is right when it is from the expected document and page."""
    return (
        chunk.metadata_.get("source") == source
        and getattr(chunk, "page", None) in pages
    )


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def count_facts(context: list[tuple], facts: list[str]) -> int:
    """How many of the answer's key phrases appear in the chunks the model reads.

    Case and spacing are ignored. This is the check that sees the "answer
    continues in the next chunk" problem: the right page can be retrieved and
    still leave out the sentence that answers the question.
    """
    text = _normalise(" ".join(chunk.content for chunk, _ in context))
    return sum(_normalise(fact) in text for fact in facts)


def score_item(
    results: list[tuple],
    source: str,
    pages: list[int],
    max_distance: float,
    context: list[tuple] | None = None,
    facts: list[str] | None = None,
) -> dict:
    """Score one answerable question from its ranked (chunk, distance) results.

    `context` is what the model would be given (the hits that pass the distance
    gate, widened if neighbours are on); `facts` are phrases its answer needs.
    """
    hits = [is_hit(chunk, source, pages) for chunk, _ in results]
    rank = next((r for r, hit in enumerate(hits, 1) if hit), None)
    hit_distances = [d for (_, d), hit in zip(results, hits, strict=True) if hit]
    best_hit = min(hit_distances) if hit_distances else None
    row = {
        "rank": rank,
        "best_hit_distance": best_hit,
        "top_distance": results[0][1] if results else None,
        "rr": hit_reciprocal_rank(hits),
        # a right-page chunk that gets past the distance gate and into the prompt
        "reaches_prompt": best_hit is not None and best_hit <= max_distance,
    }
    for k in SCORED_K_VALUES:
        row[f"recall@{k}"] = hit_recall_at_k(hits, k)
    if context is not None:
        row["context_hit"] = any(is_hit(chunk, source, pages) for chunk, _ in context)
        row["context_chars"] = sum(len(chunk.content) for chunk, _ in context)
        if facts:
            row["facts_found"] = count_facts(context, facts)
            row["facts_total"] = len(facts)
    return row


def score_off_topic(results: list[tuple], max_distance: float) -> dict:
    top = results[0][1] if results else None
    return {"top_distance": top, "refused": top is None or top > max_distance}


def summarise(rows: list[dict], off_topic: list[dict]) -> dict:
    n = len(rows)
    summary: dict = {"questions": n}
    for k in SCORED_K_VALUES:
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
    summary["mrr"] = sum(r["rr"] for r in rows) / n
    summary["reaches_prompt"] = sum(r["reaches_prompt"] for r in rows) / n
    if rows and "context_hit" in rows[0]:
        summary["context_hit"] = sum(r["context_hit"] for r in rows) / n
        chars = [r["context_chars"] for r in rows]
        summary["context_chars_mean"] = sum(chars) / n
        summary["context_chars_max"] = max(chars)
        summary["context_tokens_max_est"] = int(max(chars) / CHARS_PER_TOKEN)
    scored = [r for r in rows if "facts_total" in r]
    if scored:
        summary["fact_coverage"] = sum(
            r["facts_found"] / r["facts_total"] for r in scored
        ) / len(scored)
        summary["all_facts"] = sum(
            r["facts_found"] == r["facts_total"] for r in scored
        ) / len(scored)
    if off_topic:
        summary["off_topic_refused"] = sum(r["refused"] for r in off_topic) / len(
            off_topic
        )
    return summary


def threshold_sweep(
    rows: list[dict], off_topic: list[dict], low: float = 0.30, high: float = 0.80
) -> list[dict]:
    """For each candidate MAX_DISTANCE: how many right pages get through, and
    how many off-topic questions are still refused."""
    sweep = []
    steps = round((high - low) / 0.02)
    for i in range(steps + 1):
        ceiling = round(low + 0.02 * i, 2)
        reach = sum(
            r["best_hit_distance"] is not None and r["best_hit_distance"] <= ceiling
            for r in rows
        )
        refused = sum(
            r["top_distance"] is None or r["top_distance"] > ceiling for r in off_topic
        )
        sweep.append(
            {
                "max_distance": ceiling,
                "reaches_prompt": reach / len(rows),
                "off_topic_refused": refused / len(off_topic) if off_topic else None,
            }
        )
    return sweep


def best_ceiling(sweep: list[dict]) -> dict | None:
    """A MAX_DISTANCE that refuses every off-topic question and lets through as
    many right pages as possible; None if no ceiling does both.

    Many ceilings can tie (anything between the worst right answer and the
    closest off-topic question), so the middle of that range is returned:
    the most room for error on both sides. `low` and `high` give the range.
    """
    safe = [s for s in sweep if s["off_topic_refused"] in (1.0, None)]
    if not safe:
        return None
    most = max(s["reaches_prompt"] for s in safe)
    if most == 0:
        return None
    tied = [s for s in safe if s["reaches_prompt"] == most]
    low, high = tied[0]["max_distance"], tied[-1]["max_distance"]
    return {
        "max_distance": round((low + high) / 2, 2),
        "reaches_prompt": most,
        "off_topic_refused": tied[0]["off_topic_refused"],
        "low": low,
        "high": high,
    }


def print_report(
    per_item: list[dict], off_topic: list[dict], summary: dict, label: str
):
    print()
    print(label)
    print(
        f"{'question':<34} {'rank':>4} {'hit dist':>8}  r@1 r@3 r@5  prompt  ctx  facts"
    )
    print("-" * 88)
    for r in per_item:
        rank = r["rank"] if r["rank"] is not None else "-"
        dist = (
            f"{r['best_hit_distance']:.3f}"
            if r["best_hit_distance"] is not None
            else "-"
        )
        ctx = ""
        if "context_hit" in r:
            ctx = "yes" if r["context_hit"] else "NO"
        facts = ""
        if "facts_total" in r:
            facts = f"{r['facts_found']}/{r['facts_total']}"
            if r["facts_found"] < r["facts_total"]:
                facts += " <- missing"
        print(
            f"{r['id'][:33]:<34} {rank!s:>4} {dist:>8}   "
            f"{r['recall@1']:.0f}   {r['recall@3']:.0f}   {r['recall@5']:.0f}   "
            f"{'yes' if r['reaches_prompt'] else 'NO':>5}  {ctx:<3}  {facts}"
        )
    print("-" * 88)
    print(
        f"recall@1={summary['recall@1']:.3f}  recall@3={summary['recall@3']:.3f}  "
        f"recall@5={summary['recall@5']:.3f}  MRR={summary['mrr']:.3f}  "
        f"reaches_prompt={summary['reaches_prompt']:.3f}  ({summary['questions']} questions)"
    )
    if "fact_coverage" in summary:
        print(
            f"answer facts in the context the model reads: "
            f"coverage={summary['fact_coverage']:.3f}  "
            f"questions with every fact={summary['all_facts']:.3f}"
        )
    if "context_hit" in summary:
        print(
            f"context_hit={summary['context_hit']:.3f}  "
            f"context size: mean {summary['context_chars_mean']:.0f} chars, "
            f"max {summary['context_chars_max']} chars "
            f"(~{summary['context_tokens_max_est']} tokens; LLM_NUM_CTX must exceed this "
            f"plus ~400 for instructions)"
        )

    if off_topic:
        print()
        print("off-topic questions (must be refused):")
        for r in off_topic:
            top = f"{r['top_distance']:.3f}" if r["top_distance"] is not None else "-"
            print(
                f"  {r['id']:<26} closest chunk {top}  "
                f"{'refused' if r['refused'] else 'ANSWERED - gate too loose'}"
            )


def print_threshold(sweep: list[dict], current: float) -> None:
    print()
    print(f"MAX_DISTANCE sweep (current setting {current}):")
    print("  ceiling  right pages let through  off-topic refused")
    previous = None
    for s in sweep:
        key = (s["reaches_prompt"], s["off_topic_refused"])
        if key == previous:
            continue
        previous = key
        refused = (
            "n/a" if s["off_topic_refused"] is None else f"{s['off_topic_refused']:.2f}"
        )
        mark = "  <- current" if abs(s["max_distance"] - current) < 0.01 else ""
        print(
            f"  {s['max_distance']:.2f}     {s['reaches_prompt']:.2f}                     {refused}{mark}"
        )
    best = best_ceiling(sweep)
    if best is None:
        print(
            "  no ceiling both lets right pages through and refuses every "
            "off-topic question; fix retrieval first"
        )
    else:
        print(
            f"  suggested MAX_DISTANCE: {best['max_distance']:.2f} "
            f"(anything from {best['low']:.2f} to {best['high']:.2f} lets "
            f"{best['reaches_prompt']:.0%} of right pages through and refuses all "
            f"off-topic questions)"
        )


async def run(
    retriever: str,
    reranker: str,
    neighbors: int,
    max_chunks: int,
    save: Path | None,
    min_recall5: float | None,
) -> int:
    golden = load_golden()
    source = golden["source"]
    settings = get_settings()
    ollama = OllamaClient()
    active_reranker = build_reranker(reranker, settings.reranker_model)
    per_item: list[dict] = []
    off_topic: list[dict] = []

    try:
        async with SessionFactory() as session:
            present = await session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.metadata_["source"].astext == source)
            )
            if not present:
                print(
                    f"ERROR: no chunks from {source!r} in the database. "
                    f"Ingest it first: python scripts\\ingest_pdfs.py data\\aviation"
                )
                return 2

            async def ask(question: str) -> list[tuple]:
                return await retrieve(
                    session,
                    ollama,
                    question,
                    RETRIEVE_K,
                    hybrid=(retriever == "hybrid"),
                    reranker=active_reranker,
                )

            for item in golden["items"]:
                results = await ask(item["question"])
                # what the model would read: the hits that pass the gate,
                # widened exactly as the API does it
                context = [
                    (c, d)
                    for c, d in results[: settings.top_k]
                    if d <= settings.max_distance
                ]
                if neighbors:
                    context = await expand_with_neighbours(
                        session, context, neighbors, max_chunks
                    )
                row = score_item(
                    results,
                    source,
                    item["pages"],
                    settings.max_distance,
                    context,
                    item.get("facts"),
                )
                per_item.append({"id": item["id"], "question": item["question"], **row})

            for item in golden.get("off_topic", []):
                results = await ask(item["question"])
                off_topic.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        **score_off_topic(results, settings.max_distance),
                    }
                )
    finally:
        await ollama.close()

    summary = summarise(per_item, off_topic)
    sweep = threshold_sweep(per_item, off_topic)
    label = (
        f"retriever={retriever}  reranker={reranker}  neighbors={neighbors}  "
        f"max_distance={settings.max_distance}  top_k={settings.top_k}"
    )
    print_report(per_item, off_topic, summary, label)
    print_threshold(sweep, settings.max_distance)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps(
                {
                    "label": label,
                    "summary": summary,
                    "items": per_item,
                    "off_topic": off_topic,
                    "sweep": sweep,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nsaved {save}")

    if min_recall5 is not None and summary["recall@5"] < min_recall5:
        print(f"\nFAIL: recall@5 {summary['recall@5']:.3f} < {min_recall5}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval eval on real documents.")
    parser.add_argument("--retriever", choices=["vector", "hybrid"], default="vector")
    parser.add_argument("--reranker", choices=["none", "crossencoder"], default="none")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=0,
        help="add this many following chunks after each hit before scoring the context",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=get_settings().context_max_chunks,
        help="most chunks in one prompt, hits included (default: CONTEXT_MAX_CHUNKS)",
    )
    parser.add_argument(
        "--save", type=Path, help="write the full results to this JSON file"
    )
    parser.add_argument(
        "--min-recall5",
        type=float,
        help="exit 1 if mean recall@5 is below this (for use as a gate)",
    )
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            run(
                args.retriever,
                args.reranker,
                args.neighbors,
                args.max_chunks,
                args.save,
                args.min_recall5,
            )
        )
    )
