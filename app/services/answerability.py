"""Can the stored documents answer this question at all?

The distance gate asks "is some chunk close to the question?". That is a topic
test, not an answerability test: "how do I repair a carbon fiber propeller blade"
sits close to a manual that covers propellers and carbon steel, although the
manual says nothing about carbon fiber. This module adds a blunt but
deterministic second test: a distinctive word in the question ("fiber", "Airbus",
"Garmin") that occurs in *no* stored chunk is evidence the documents cannot
answer it.

It is a full-text lookup against the existing `tsv` GIN index, so it needs no
model call. Words are matched the way the keyword search matches them (English
stemming), plus a small British/American spelling table so "colour" still finds
"color". It cannot catch a question whose every word appears somewhere in the
corpus (that is what the distance gate and, later, an answer check are for).
"""

import re
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Words that carry no subject matter, however they are spelled. Kept short on
# purpose: anything a manual really contains is found in the corpus anyway, so
# this list only has to stop everyday question words from being tested.
STOPWORDS = frozenset(
    [
        "about",
        "above",
        "after",
        "again",
        "against",
        "also",
        "always",
        "among",
        "another",
        "answer",
        "anyone",
        "anything",
        "around",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "bring",
        "cannot",
        "check",
        "could",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "either",
        "enough",
        "ever",
        "every",
        "find",
        "first",
        "from",
        "further",
        "gets",
        "give",
        "given",
        "goes",
        "going",
        "have",
        "having",
        "here",
        "into",
        "itself",
        "just",
        "keep",
        "know",
        "large",
        "last",
        "later",
        "least",
        "less",
        "like",
        "list",
        "long",
        "look",
        "made",
        "make",
        "makes",
        "many",
        "might",
        "more",
        "most",
        "much",
        "must",
        "near",
        "need",
        "never",
        "next",
        "none",
        "only",
        "other",
        "over",
        "please",
        "quite",
        "rather",
        "really",
        "right",
        "same",
        "several",
        "shall",
        "should",
        "show",
        "since",
        "small",
        "some",
        "still",
        "such",
        "sure",
        "take",
        "tell",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "this",
        "those",
        "through",
        "today",
        "under",
        "until",
        "upon",
        "used",
        "uses",
        "using",
        "very",
        "want",
        "well",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "whilst",
        "will",
        "with",
        "within",
        "without",
        "would",
        "wrong",
        "your",
        "yours",
        "explain",
        "describe",
        "procedure",
    ]
)

# British -> American spellings the aviation literature mixes freely. The check
# tries both, so a spelling difference never makes a word "unknown".
SPELLING_VARIANTS = {
    "aluminium": "aluminum",
    "analyse": "analyze",
    "centre": "center",
    "colour": "color",
    "fibre": "fiber",
    "grey": "gray",
    "kilometre": "kilometer",
    "litre": "liter",
    "metre": "meter",
    "millimetre": "millimeter",
    "tyre": "tire",
    "sulphur": "sulfur",
}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Answerability:
    """Result of the vocabulary check for one question."""

    terms: dict[str, int] = field(default_factory=dict)  # term -> chunks containing it
    unknown: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unknown


def question_terms(question: str) -> list[str]:
    """Distinctive words of a question, lower-cased, in order, without repeats.

    Dropped: stopwords, plain numbers (a value the question mentions is not
    vocabulary), and words shorter than four letters unless they mix letters and
    digits ("g1000", "ge90", "a320" are exactly the words that matter).
    """
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(question.lower()):
        has_digit = any(ch.isdigit() for ch in token)
        has_alpha = any(ch.isalpha() for ch in token)
        if not has_alpha:
            continue
        if token in STOPWORDS:
            continue
        if len(token) < 4 and not has_digit:
            continue
        seen.setdefault(token, None)
    return list(seen)


def _alternatives(term: str) -> list[str]:
    """The term plus its other spelling, so either one counts as a match."""
    alts = [term]
    american = SPELLING_VARIANTS.get(term)
    if american:
        alts.append(american)
    else:
        for british, us in SPELLING_VARIANTS.items():
            if term == us:
                alts.append(british)
    # colour, coloured, honour... <-> color, colored, honor...
    if "our" in term and not term.endswith(("hour", "hours", "four", "pour", "tour")):
        alts.append(term.replace("our", "or"))
    # -ise / -isation words: "recognise" <-> "recognize"
    if term.endswith("ise") and len(term) > 5:
        alts.append(term[:-3] + "ize")
    elif term.endswith("ize") and len(term) > 5:
        alts.append(term[:-3] + "ise")
    return alts


async def term_chunk_counts(session: AsyncSession, terms: list[str]) -> dict[str, int]:
    """How many stored chunks contain each term (0 = nowhere in the corpus).

    One round trip: the terms are unnested server-side and each is counted with
    the GIN-indexed `tsv @@ to_tsquery(...)`. A term that the English text-search
    config discards entirely (a stopword) is left out of the result, so it can
    never be reported as unknown.
    """
    if not terms:
        return {}
    queries = [" | ".join(_alternatives(t)) for t in terms]
    rows = await session.execute(
        text(
            """
            SELECT q.term,
                   (SELECT count(*) FROM document_chunks c
                     WHERE c.tsv @@ to_tsquery('english', q.tsq)) AS n
            FROM unnest(CAST(:terms AS text[]), CAST(:queries AS text[]))
                 AS q(term, tsq)
            WHERE numnode(to_tsquery('english', q.tsq)) > 0
            """
        ),
        {"terms": terms, "queries": queries},
    )
    return {term: int(n) for term, n in rows.all()}


def judge(counts: dict[str, int], min_chunks: int = 1) -> Answerability:
    """Unknown = fewer than `min_chunks` chunks contain the term."""
    unknown = [t for t, n in counts.items() if n < min_chunks]
    return Answerability(terms=counts, unknown=unknown)


async def check_answerability(
    session: AsyncSession, question: str, min_chunks: int = 1
) -> Answerability:
    counts = await term_chunk_counts(session, question_terms(question))
    return judge(counts, min_chunks)
