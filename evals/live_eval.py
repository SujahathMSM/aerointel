"""Live eval of the trust layer: needs the real model, so it runs on your machine.

`doc_eval` measures retrieval and never calls the language model. Three things
can only be measured with it:

1. The answer check (ANSWER_CHECK): for each question that passes the distance
   gate, the model is asked "do these passages answer it?". On right questions
   a NO is a wrong refusal; on off-topic questions a NO is a catch.
2. Prompt rule 2 ("if the context does not answer, say so"): for off-topic
   questions the gate lets through, does the written answer decline?
3. Figure verification: for right questions, does the check flag numbers in an
   answer to a question the manual does answer? Each flag is a candidate false
   alarm; the answers are saved so they can be read.

    python -m evals.live_eval --retriever hybrid
    python -m evals.live_eval --retriever hybrid --context-max-distance 0.52
    python -m evals.live_eval --only weld-metal-fuel-tank --only b737-jacking
    python -m evals.live_eval --check-only            # skip writing answers (faster)
    python -m evals.live_eval --save evals/results/live.json

It uses the same retrieval, gate, reading limit, prompts and checks as the API,
whatever ANSWER_CHECK and VERIFY_MODE are set to: everything is measured, the
modes only decide what the API does with the results. Needs Postgres with the
manual ingested and Ollama with both models.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.db.base import SessionFactory
from app.llm.ollama import OllamaClient
from app.services.generation import (
    CHECK_NO,
    build_prompt,
    check_answerable,
    context_block,
    select_context,
    system_prompt,
)
from app.services.pipeline import retrieve
from app.services.verification import verify_answer
from evals.doc_eval import count_facts, is_hit, load_golden

# Ways the model says the passages do not answer the question. Read the saved
# answers before trusting the count: this is a heuristic, not a judge.
_DECLINE = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not)\s+"
    r"(?:specify|state|provide|mention|cover|contain|include|say|give|address|list)"
    r"|\bnot\s+(?:specified|stated|provided|mentioned|covered|included|given|"
    r"available|addressed|listed|found)\b"
    r"|\bno\s+(?:information|mention|details|data|specific)\b"
    r"|\bcan(?:not|'t)\s+be\s+determined\b"
    r"|\bunable to (?:determine|answer|find)\b"
    r"|\bnot (?:in|part of) the (?:context|provided)",
    re.IGNORECASE,
)


def looks_like_decline(answer: str) -> bool:
    """Does the answer say the passages do not answer the question?"""
    return bool(_DECLINE.search(answer))


def summarise_live(right: list[dict], off_topic: list[dict]) -> dict:
    """Counts over the rows `run` produces (pure, so it is unit-tested)."""

    def passed(rows):
        return [r for r in rows if r["gate_passed"]]

    right_in, off_in = passed(right), passed(off_topic)
    answered_right = [r for r in right_in if r.get("answer") is not None]
    answered_off = [r for r in off_in if r.get("answer") is not None]
    return {
        "right_total": len(right),
        "right_passed_gate": len(right_in),
        "right_check_no": sum(r["check"] == CHECK_NO for r in right_in),
        "right_check_unclear": sum(r["check"] == "unclear" for r in right_in),
        "right_answered": len(answered_right),
        "right_declined": sum(r["declined"] for r in answered_right),
        "right_flagged": sum(not r["verification_ok"] for r in answered_right),
        "right_with_numbers": sum(r["numbers_checked"] > 0 for r in answered_right),
        "right_all_facts_in_answer": sum(
            r["facts_in_answer"] == r["facts_total"]
            for r in answered_right
            if r["facts_total"]
        ),
        "off_total": len(off_topic),
        "off_passed_gate": len(off_in),
        "off_check_no": sum(r["check"] == CHECK_NO for r in off_in),
        "off_answered": len(answered_off),
        "off_declined": sum(r["declined"] for r in answered_off),
        "off_by_group": {
            group: _off_counts([r for r in off_topic if r.get("group") == group])
            for group in sorted({r.get("group") or "-" for r in off_topic} - {"-"})
        },
    }


def _off_counts(rows: list[dict]) -> dict:
    """Gate, check and decline counts for one group of off-topic questions."""
    gate_in = [r for r in rows if r["gate_passed"]]
    answered = [r for r in gate_in if r.get("answer") is not None]
    return {
        "total": len(rows),
        "passed_gate": len(gate_in),
        "check_no": sum(r["check"] == CHECK_NO for r in gate_in),
        "answered": len(answered),
        "declined": sum(r["declined"] for r in answered),
    }


def print_live(right: list[dict], off_topic: list[dict], summary: dict) -> None:
    print()
    print(f"{'question':<30} {'gate':>6} {'check':>8} {'answer':>9} {'figures':>16}")
    print("-" * 74)
    for r in [*right, *off_topic]:
        gate = "pass" if r["gate_passed"] else "refuse"
        check = r["check"] or "-"
        if r.get("answer") is None:
            answer = "-"
        else:
            answer = "declined" if r["declined"] else "answered"
        if r.get("answer") is None or r["numbers_checked"] == 0:
            figures = "-"
        elif r["verification_ok"]:
            figures = f"ok ({r['numbers_checked']})"
        else:
            figures = "FLAG " + ",".join(r["unsupported"])[:10]
        tag = ""
        if r["kind"] == "off_topic":
            tag = f"  ({(r.get('group') or 'off-topic').replace('_', '-')})"
        print(f"{r['id']:<30} {gate:>6} {check:>8} {answer:>9} {figures:>16}{tag}")
    s = summary
    print("-" * 74)
    print(
        f"right questions: {s['right_passed_gate']}/{s['right_total']} pass the gate; "
        f"answer check says NO to {s['right_check_no']} (would be wrongly refused), "
        f"unclear {s['right_check_unclear']}"
    )
    if s["right_answered"]:
        print(
            f"  answers: {s['right_declined']}/{s['right_answered']} decline "
            f"(missed answers); figures flagged in {s['right_flagged']} "
            f"(of {s['right_with_numbers']} with numbers; read them: false alarm or "
            f"misquote?); every fact phrase in {s['right_all_facts_in_answer']}"
        )
    print(
        f"off-topic: {s['off_passed_gate']}/{s['off_total']} pass the gate; "
        f"answer check says NO to {s['off_check_no']} of those (caught)"
    )
    if s["off_answered"]:
        print(
            f"  answers: {s['off_declined']}/{s['off_answered']} decline "
            f"(prompt rule 2 obeyed)"
        )
    for group, c in s.get("off_by_group", {}).items():
        line = (
            f"  {group.replace('_', '-')}: {c['passed_gate']}/{c['total']} pass the "
            f"gate, check NO to {c['check_no']}"
        )
        if c["answered"]:
            line += f", {c['declined']}/{c['answered']} answers decline"
        print(line)


async def run(
    retriever: str,
    context_max_distance: float | None,
    check_only: bool,
    only: set[str] | None,
    save: Path | None,
) -> int:
    golden = load_golden()
    source = golden["source"]
    settings = get_settings()
    ollama = OllamaClient()
    system = system_prompt()
    right: list[dict] = []
    off_topic: list[dict] = []
    started = time.monotonic()

    async def one(item: dict, kind: str) -> dict:
        async with SessionFactory() as session:
            served = await retrieve(
                session,
                ollama,
                item["question"],
                settings.top_k,
                hybrid=(retriever == "hybrid"),
            )
            await session.commit()
        context = select_context(served, settings.max_distance, context_max_distance)
        row: dict = {
            "id": item["id"],
            "kind": kind,
            # off-topic subgroup from the golden file: generic | outside | near_topic
            "group": item.get("kind"),
            "question": item["question"],
            "gate_passed": bool(context),
            "read_right_page": bool(
                item.get("pages")
                and any(is_hit(c, source, item["pages"]) for c, _ in context)
            ),
            "check": None,
            "answer": None,
            "declined": None,
            "verification_ok": None,
            "numbers_checked": 0,
            "unsupported": [],
            "facts_total": len(item.get("facts", [])),
            "facts_in_answer": 0,
        }
        if not context:
            return row
        row["check"] = await check_answerable(ollama, item["question"], context)
        if check_only:
            return row
        answer = await ollama.chat(system.text, build_prompt(item["question"], context))
        verification = verify_answer(
            answer, [context_block(c) for c, _ in context], item["question"]
        )
        row.update(
            answer=answer,
            declined=looks_like_decline(answer),
            verification_ok=verification.ok,
            numbers_checked=verification.checked,
            unsupported=[*verification.unsupported, *verification.unit_mismatch],
            facts_in_answer=count_facts([(_Text(answer), 0.0)], item.get("facts", [])),
        )
        return row

    try:
        for item in golden["items"]:
            if only is None or item["id"] in only:
                right.append(await one(item, "right"))
                print(f"  {item['id']}", flush=True)
        for item in golden.get("off_topic", []):
            if only is None or item["id"] in only:
                off_topic.append(await one(item, "off_topic"))
                print(f"  {item['id']}", flush=True)
    finally:
        await ollama.close()

    summary = summarise_live(right, off_topic)
    label = (
        f"retriever={retriever}  max_distance={settings.max_distance}  "
        f"context_max_distance={context_max_distance}  top_k={settings.top_k}  "
        f"model={settings.llm_model}  prompt={system.label}  "
        f"{time.monotonic() - started:.0f}s"
    )
    print()
    print(label)
    print_live(right, off_topic, summary)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps(
                {
                    "label": label,
                    "summary": summary,
                    "right": right,
                    "off_topic": off_topic,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nsaved {save} (the full answers are in it)")
    return 0


class _Text:
    """Lets `count_facts` read an answer as if it were a chunk."""

    def __init__(self, content: str):
        self.content = content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live eval of the trust layer.")
    parser.add_argument("--retriever", choices=["vector", "hybrid"], default="vector")
    parser.add_argument(
        "--context-max-distance",
        type=float,
        default=get_settings().context_max_distance,
        help="reading limit past the gate (default: CONTEXT_MAX_DISTANCE)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only run the answer check; do not write answers",
    )
    parser.add_argument(
        "--only", action="append", metavar="ID", help="run only this question id"
    )
    parser.add_argument("--save", type=Path, help="write all rows and answers here")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            run(
                args.retriever,
                args.context_max_distance,
                args.check_only,
                set(args.only) if args.only else None,
                args.save,
            )
        )
    )
