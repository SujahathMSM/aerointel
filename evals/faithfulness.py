"""Faithfulness: does the answer only claim what the CONTEXT actually says?

Two tiers, deliberately separate, because they fail differently.

1. Citation validity — deterministic, free, always on. Every `[n]` the
   answer cites must point at a CONTEXT block that was actually supplied.
   A model that cites [4] when it was handed three blocks has invented a
   source, and no language model is needed to catch that. Cheap checks
   that cannot be wrong are worth more than expensive ones that can.

2. LLM-as-judge — opt-in, costs a second model call per answer. A separate
   model reads (context, answer) and rules on whether every claim is
   supported. This catches the harder case: a fluent answer, correctly
   cited, that quietly adds a figure the context never gave.

The judge is itself a fallible model, so its verdict is evidence, not
proof. Treat a drop in judge faithfulness as a prompt to go read the
answers — not as a number to optimise blindly. Citation validity, being
deterministic, is the one safe to gate on.
"""

import re

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

VERDICT_LINE = re.compile(r"VERDICT:\s*(SUPPORTED|UNSUPPORTED)", re.IGNORECASE)


def cited_indices(answer: str) -> set[int]:
    """Every block number the answer cites, from [1] and [2, 3] alike."""
    found: set[int] = set()
    for group in CITATION.findall(answer):
        for part in group.split(","):
            found.add(int(part.strip()))
    return found


def citation_validity(answer: str, n_context_blocks: int) -> float:
    """1.0 if every cited block exists, else the fraction that do.

    An answer citing nothing scores 0.0: the grounding contract requires
    citations, so their absence is a failure of the contract, not a
    neutral outcome. (A refusal never reaches this function — no answer
    was generated.)
    """
    cited = cited_indices(answer)
    if not cited:
        return 0.0
    valid = {index for index in cited if 1 <= index <= n_context_blocks}
    return len(valid) / len(cited)


def has_hallucinated_citation(answer: str, n_context_blocks: int) -> bool:
    """True if the answer points at a block that was never supplied."""
    return any(
        index < 1 or index > n_context_blocks for index in cited_indices(answer)
    )


def build_judge_input(question: str, context: str, answer: str) -> str:
    """The user-side payload for the judge; instructions live in the registry."""
    return (
        f"CONTEXT:\n\n{context}\n\n---\n\n"
        f"QUESTION: {question}\n\n---\n\n"
        f"ANSWER: {answer}\n\n---\n\nGRADE:"
    )


def parse_judge_verdict(reply: str) -> float | None:
    """1.0 SUPPORTED, 0.0 UNSUPPORTED, None if the judge was unparseable.

    None is deliberately not 0.0. A judge that returned garbage tells you
    nothing about the answer, and silently scoring it as a hallucination
    would poison the metric with the judge's own failures.
    """
    match = VERDICT_LINE.search(reply or "")
    if match is None:
        return None
    return 1.0 if match.group(1).upper() == "SUPPORTED" else 0.0


def mean_or_none(values: list[float | None]) -> float | None:
    """Mean over the scores that exist; None if every one was unparseable."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)
