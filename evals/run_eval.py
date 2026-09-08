"""Run the retrieval eval: golden set -> live pipeline -> metrics.

This is the Phase 1 scoreboard. Run it before and after any retrieval
change and compare the numbers:

    python -m evals.run_eval                                  # baseline
    python -m evals.run_eval --retriever hybrid               # + BM25/RRF
    python -m evals.run_eval --reranker crossencoder          # + rerank
    python -m evals.run_eval --retriever hybrid --reranker crossencoder
    python -m evals.run_eval --faithfulness                   # + hallucination check

Retrieval metrics (recall@k, MRR) only need the embedding model.
--faithfulness additionally generates an answer per question and grades it:
citation validity is deterministic and always reported, and an LLM judge
rules on whether every claim is supported by the retrieved context. That
costs two extra model calls per question, so it is opt-in.

Requires: Postgres up (with sample data loaded via scripts/load_sample_data.py)
and Ollama running. The cross-encoder additionally needs
requirements-rerank.txt installed. Exits 1 if mean recall@5 drops below
the threshold, or if any answer cites a context block that does not exist.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.base import SessionFactory
from app.db.models import DocumentChunk
from app.llm.ollama import OllamaClient
from app.prompts import get_prompt
from app.services.generation import (
    PROMPT_NAME,
    build_prompt,
    system_prompt,
)
from app.services.pipeline import retrieve
from app.services.reranking import build_reranker
from evals.faithfulness import (
    build_judge_input,
    citation_validity,
    has_hallucinated_citation,
    mean_or_none,
    parse_judge_verdict,
)
from evals.metrics import aggregate, recall_at_k, reciprocal_rank

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"

# Fetch deeper than we score, so we can see "how far down" a miss landed.
RETRIEVE_K = 10
SCORED_K_VALUES = (1, 3, 5)

# The gate: if mean recall@5 falls below this, the eval fails (exit 1).
MIN_RECALL_AT_5 = 0.9

# Judge prompt lives in the same versioned registry as the answer prompt, so
# a change to how answers are graded is as traceable as a change to how they
# are produced.
JUDGE_PROMPT_NAME = "faithfulness_judge"


def load_golden_set() -> list[dict]:
    items = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    if not items:
        raise RuntimeError(f"{GOLDEN_SET_PATH} is empty")
    return items


async def resolve_expected_ids(
    session, items: list[dict]
) -> dict[tuple[str, str], int]:
    """Map each golden item's (source, section) to a chunk id in the DB.

    First match wins; the sample corpus has unique (source, section) pairs.
    A missing key means the corpus is not loaded — say so clearly instead
    of silently scoring zeros.
    """
    rows = (await session.execute(select(DocumentChunk))).scalars().all()
    lookup: dict[tuple[str, str], int] = {}
    for chunk in rows:
        key = (
            str(chunk.metadata_.get("source", "")),
            str(chunk.metadata_.get("section", "")),
        )
        lookup.setdefault(key, chunk.id)
    return lookup


async def score_faithfulness(
    ollama, question: str, results: list[tuple]
) -> tuple[str, float, float | None]:
    """Generate an answer for one question and grade it.

    Returns (answer, citation_validity, judge_score). judge_score is None
    when the judge's reply could not be parsed — that is a fact about the
    judge, not about the answer, and is kept out of the mean.

    The answer is generated exactly the way /v1/query generates it: same
    distance gate, same numbered CONTEXT blocks, same registry prompt. An
    eval that grades a differently-built answer grades nothing useful.
    """
    settings = get_settings()
    relevant = [(c, d) for c, d in results if d <= settings.max_distance]
    if not relevant:
        # Structural refusal: no answer exists to grade. Not a failure —
        # it is the system working, so it is excluded from the means.
        return "", float("nan"), None

    answer_prompt = system_prompt()
    answer = await ollama.chat(answer_prompt.text, build_prompt(question, relevant))

    context = build_prompt(question, relevant)
    judge_prompt = get_prompt(JUDGE_PROMPT_NAME)
    verdict = await ollama.chat(
        judge_prompt.text, build_judge_input(question, context, answer)
    )

    return (
        answer,
        citation_validity(answer, len(relevant)),
        parse_judge_verdict(verdict),
    )


def print_report(per_item: list[dict], summary: dict, run_label: str) -> None:
    print()
    print(run_label)
    print(f"{'question':<58} {'rank':>4} {'dist':>7}  r@1  r@3  r@5")
    print("-" * 88)
    for row in per_item:
        rank = row["rank"] if row["rank"] is not None else "-"
        distance = f"{row['distance']:.4f}" if row["distance"] is not None else "-"
        print(
            f"{row['question'][:57]:<58} {rank!s:>4} {distance:>7}"
            f"  {row['recall@1']:.0f}    {row['recall@3']:.0f}    {row['recall@5']:.0f}"
        )
    print("-" * 88)
    print(
        f"recall@1={summary['recall@1']:.3f}  "
        f"recall@3={summary['recall@3']:.3f}  "
        f"recall@5={summary['recall@5']:.3f}  "
        f"MRR={summary['mrr']:.3f}"
    )

    graded = [row for row in per_item if "citation_validity" in row]
    answered = [row for row in graded if row.get("answer")]
    if graded:
        cite_mean = mean_or_none([row["citation_validity"] for row in answered])
        judge_mean = mean_or_none([row["judge"] for row in answered])
        refusals = len(graded) - len(answered)
        unparsed = sum(1 for row in answered if row["judge"] is None)
        print(
            f"citation_validity="
            f"{'n/a' if cite_mean is None else format(cite_mean, '.3f')}  "
            f"judge_faithfulness="
            f"{'n/a' if judge_mean is None else format(judge_mean, '.3f')}  "
            f"answered={len(answered)}/{len(graded)}"
            + (f"  refused={refusals}" if refusals else "")
            + (f"  judge_unparseable={unparsed}" if unparsed else "")
        )


async def run(
    retriever: str = "vector",
    reranker: str = "none",
    faithfulness: bool = False,
) -> int:
    items = load_golden_set()
    ollama = OllamaClient()
    active_reranker = build_reranker(reranker, get_settings().reranker_model)

    try:
        async with SessionFactory() as session:
            lookup = await resolve_expected_ids(session, items)

            per_item: list[dict] = []
            for item in items:
                key = (
                    str(item["expected"].get("source", "")),
                    str(item["expected"].get("section", "")),
                )
                expected_id = lookup.get(key)
                if expected_id is None:
                    print(
                        f"ERROR: no chunk in the database matches "
                        f"source={key[0]!r}, section={key[1]!r}. "
                        f"Load the sample corpus first: "
                        f"python scripts/load_sample_data.py"
                    )
                    return 2

                results = await retrieve(
                    session,
                    ollama,
                    item["question"],
                    RETRIEVE_K,
                    hybrid=(retriever == "hybrid"),
                    reranker=active_reranker,
                )
                ranked_ids = [chunk.id for chunk, _ in results]
                distances = {chunk.id: d for chunk, d in results}

                rank = next(
                    (r for r, cid in enumerate(ranked_ids, 1) if cid == expected_id),
                    None,
                )
                row = {
                    "question": item["question"],
                    "rank": rank,
                    "distance": distances.get(expected_id),
                    "rr": reciprocal_rank(ranked_ids, expected_id),
                }
                for k in SCORED_K_VALUES:
                    row[f"recall@{k}"] = recall_at_k(ranked_ids, expected_id, k)

                if faithfulness:
                    answer, cite_score, judge_score = await score_faithfulness(
                        ollama, item["question"], results
                    )
                    n_blocks = len(
                        [c for c, d in results if d <= get_settings().max_distance]
                    )
                    row["answer"] = answer
                    row["citation_validity"] = cite_score
                    row["judge"] = judge_score
                    row["bad_citation"] = bool(answer) and has_hallucinated_citation(
                        answer, n_blocks
                    )

                per_item.append(row)
    finally:
        await ollama.close()

    summary = aggregate(per_item, SCORED_K_VALUES)
    # Name the exact configuration behind these numbers. A score with no
    # prompt version attached cannot be compared to next week's score.
    try:
        prompt_label = system_prompt().label
    except KeyError:  # pragma: no cover - only if prompt files are missing
        prompt_label = f"{PROMPT_NAME}@MISSING"
    run_label = f"retriever={retriever}  reranker={reranker}  prompt={prompt_label}"
    print_report(per_item, summary, run_label)

    bad = [row for row in per_item if row.get("bad_citation")]
    if bad:
        print(
            f"\nFAIL: {len(bad)} answer(s) cited a CONTEXT block that was "
            f"never supplied — the model invented a source:"
        )
        for row in bad:
            print(f"  - {row['question'][:70]}")
        return 1

    if summary["recall@5"] < MIN_RECALL_AT_5:
        print(
            f"\nFAIL: recall@5 {summary['recall@5']:.3f} < threshold {MIN_RECALL_AT_5}"
        )
        return 1
    print(f"\nPASS: recall@5 {summary['recall@5']:.3f} >= {MIN_RECALL_AT_5}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the retrieval eval.")
    parser.add_argument(
        "--retriever",
        choices=["vector", "hybrid"],
        default="vector",
        help="which retrieval stage to score (default: vector)",
    )
    parser.add_argument(
        "--reranker",
        choices=["none", "crossencoder"],
        default="none",
        help="reranker applied on top of retrieval (default: none)",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="also generate answers and grade them for hallucination",
    )
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            run(
                retriever=args.retriever,
                reranker=args.reranker,
                faithfulness=args.faithfulness,
            )
        )
    )
