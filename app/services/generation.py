import hashlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.answer_cache import AnswerCache, CacheHit
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.ollama import OllamaClient
from app.prompts import Prompt, get_prompt
from app.services.answerability import check_answerability
from app.services.context import expand_with_neighbours
from app.services.pipeline import retrieve
from app.services.revisions import SourceRevision, find_source_revisions
from app.services.verification import Verification, parse_citations, verify_answer
from app.services.warnings import SafetyWarning, find_warnings, mark_cited

logger = get_logger(__name__)

# Ollama's default context window when LLM_NUM_CTX is not set (tokens), and a
# deliberately pessimistic characters-per-token figure for technical English.
DEFAULT_NUM_CTX = 4096
CHARS_PER_TOKEN = 3.5

# The grounding instructions live in app/prompts/grounded_answer.<version>.txt,
# not as a literal here: a reworded prompt is then a reviewable new file, and
# every answer can name the exact version that produced it.
PROMPT_NAME = "grounded_answer"


def system_prompt() -> Prompt:
    """The active grounding prompt: pinned by settings, else newest."""
    return get_prompt(PROMPT_NAME, get_settings().prompt_version)


NO_CONTEXT_MESSAGE = (
    "No chunk in the database is close enough to this question "
    "(nothing under distance {ceiling}). Refusing to answer rather than "
    "guessing. Either the relevant document has not been ingested yet, "
    "or the question needs different wording."
)


UNKNOWN_TERMS_MESSAGE = (
    "None of the stored documents contain: {terms}. Refusing to answer rather "
    "than guessing. The relevant document may not have been ingested, or the "
    "question may use different wording."
)
UNVERIFIED_MESSAGE = (
    "The drafted answer contained figures that do not appear in the passages it "
    "cites ({figures}), so it was withheld. Read the retrieved passages directly."
)
NOT_IN_CONTEXT_MESSAGE = (
    "The closest passages are on this topic but do not answer the question, so "
    "no answer was generated. Read the retrieved passages directly, or ingest the "
    "document that covers it."
)

REASON_DISTANCE = "too_far"
REASON_UNKNOWN_TERMS = "unknown_terms"
REASON_UNVERIFIED = "unverified_numbers"
REASON_NOT_IN_CONTEXT = "not_in_context"

# The answer check's prompt (app/prompts/answer_check.<version>.txt) and the
# words its reply is read as. Anything else counts as "unclear", which never
# refuses: a check that cannot be read must not silence a real answer.
ANSWER_CHECK_PROMPT = "answer_check"
CHECK_YES = "yes"
CHECK_NO = "no"
CHECK_UNCLEAR = "unclear"


def refusal_message(
    reason: str | None, unknown_terms: list[str], figures: list[str], ceiling: float
) -> str:
    """The text that goes with a refusal, chosen by why it was refused."""
    if reason == REASON_UNKNOWN_TERMS:
        return UNKNOWN_TERMS_MESSAGE.format(terms=", ".join(unknown_terms))
    if reason == REASON_UNVERIFIED:
        return UNVERIFIED_MESSAGE.format(figures=", ".join(figures))
    if reason == REASON_NOT_IN_CONTEXT:
        return NOT_IN_CONTEXT_MESSAGE
    message = NO_CONTEXT_MESSAGE.format(ceiling=ceiling)
    if unknown_terms:
        message += f" No stored document contains: {', '.join(unknown_terms)}."
    return message


@dataclass
class AnswerResult:
    """Outcome of one grounded query."""

    question: str
    refused: bool
    answer: str | None = None
    chunks: list = field(default_factory=list)
    # Which prompt version produced `answer` (None on a refusal — no prompt
    # was used, because the model was never called).
    prompt_version: str | None = None
    # Why a refusal happened (REASON_*), None for an answer.
    refusal_reason: str | None = None
    # Question words found in no stored chunk (empty when the check is off).
    unknown_terms: list[str] = field(default_factory=list)
    # Result of checking the answer's numbers against its cited passages.
    verification: Verification | None = None
    # The model's own yes/no on "does the context answer this?" (None when off).
    answer_check: str | None = None
    # The answer came from the semantic cache, not a new model call.
    cached: bool = False
    # WARNING / CAUTION / NOTE text from the passages read (never on a refusal
    # by the gates or the answer check), cited blocks first within a level.
    warnings: list[SafetyWarning] = field(default_factory=list)
    # The documents read: revision, when stored, newer revision / superseded.
    sources: list[SourceRevision] = field(default_factory=list)


@dataclass
class PreparedAnswer:
    """Everything decided before the model is called.

    Retrieval, the distance gate and prompt assembly happen here, once, for
    both the plain and the streaming endpoint. `refused` means nothing cleared
    the ceiling: `system`/`prompt` are then None and the model must not be
    called.
    """

    question: str
    refused: bool
    chunks: list = field(default_factory=list)
    system: str | None = None
    prompt: str | None = None
    prompt_version: str | None = None
    refusal_reason: str | None = None
    unknown_terms: list[str] = field(default_factory=list)
    answer_check: str | None = None
    # The question's embedding, kept for the answer cache (None when it is off).
    query_embedding: list[float] | None = None
    # Safety labels in the passages the model reads (empty on a refusal).
    warnings: list[SafetyWarning] = field(default_factory=list)
    # Revision notes for the documents those passages come from.
    sources: list[SourceRevision] = field(default_factory=list)


def _where(chunk, issued: bool = True) -> str:
    """', page 3-3 (issued 2001-09-27, PDF page 120)' for PDF chunks; '' otherwise.

    The issue date is the one printed in the chunk's own page header (pages of
    a change-page document carry different issues). `issued=False` leaves it out.
    """
    page = getattr(chunk, "page", None)
    if page is None:
        return ""
    label = getattr(chunk, "page_label", None)
    issue = (
        (getattr(chunk, "metadata_", None) or {}).get("page_issued") if issued else None
    )
    if label:
        inner = f"issued {issue}, PDF page {page}" if issue else f"PDF page {page}"
        return f", page {label} ({inner})"
    return f", PDF page {page}" + (f" (issued {issue})" if issue else "")


def warn_if_prompt_may_not_fit(system: str, prompt: str) -> bool:
    """Log when the prompt is close to the model's context window.

    Ollama silently drops whatever does not fit, so an over-long prompt loses
    its beginning without any error. This is an estimate (no tokenizer), meant
    to catch "I turned context widening on and forgot the window".
    """
    window = get_settings().llm_num_ctx or DEFAULT_NUM_CTX
    estimated = int((len(system) + len(prompt)) / CHARS_PER_TOKEN)
    too_long = estimated > window * 0.85
    if too_long:
        logger.warning(
            "prompt may not fit the model context window",
            estimated_tokens=estimated,
            num_ctx=window,
        )
    return too_long


def context_ceiling(max_distance: float, context_max_distance: float | None) -> float:
    """The distance up to which retrieved chunks are given to the model.

    Never tighter than the refusal gate: a chunk that passes the gate is always
    read. None means "same as the gate", the original behaviour.
    """
    if context_max_distance is None:
        return max_distance
    return max(max_distance, context_max_distance)


def select_context(
    retrieved: list[tuple],
    max_distance: float,
    context_max_distance: float | None = None,
) -> list[tuple]:
    """What the model reads, in retrieval order; empty means refuse.

    Two separate questions. *Answer at all?* -- only if some retrieved chunk is
    within `max_distance` (the structural refusal, unchanged). *Read what?* --
    every retrieved chunk within the context ceiling. With the ceiling above the
    gate, an answer chunk that ranks second at 0.495 still reaches the model when
    the question passed the gate on another chunk at 0.422; a question whose
    closest chunk is 0.495 is still refused.
    """
    if not any(distance <= max_distance for _, distance in retrieved):
        return []
    ceiling = context_ceiling(max_distance, context_max_distance)
    return [(chunk, d) for chunk, d in retrieved if d <= ceiling]


def answer_check_prompt(question: str, chunks: list[tuple]) -> str:
    """The same numbered context the answer would get, then a yes/no question."""
    return build_prompt(question, chunks).removesuffix("ANSWER:") + "REPLY (YES or NO):"


def answer_check_system(version: str | None = None):
    """The answer-check prompt: `version`, else ANSWER_CHECK_VERSION, else newest."""
    return get_prompt(
        ANSWER_CHECK_PROMPT, version or get_settings().answer_check_version
    )


def parse_answer_check(reply: str) -> str:
    """YES / NO from the first word of the reply; anything else is unclear."""
    words = reply.strip().split()
    first = words[0].strip(".,:;!*\"'").lower() if words else ""
    if first == "yes":
        return CHECK_YES
    if first == "no":
        return CHECK_NO
    return CHECK_UNCLEAR


async def check_answerable(
    ollama: OllamaClient,
    question: str,
    chunks: list[tuple],
    version: str | None = None,
) -> str:
    """Ask the model whether the context answers the question (one short call).

    The distance gate measures topic, not whether the passages hold the
    answer: "torque for a helicopter main rotor blade bolt" sits as close to
    the torque table as a real torque question does. Reading the passages is
    the only way to tell, so the model is asked before it writes an answer.
    """
    prompt = answer_check_system(version)
    reply = await ollama.chat(prompt.text, answer_check_prompt(question, chunks))
    return parse_answer_check(reply)


def context_block(chunk, for_verification: bool = False) -> str:
    """One CONTEXT block as the model sees it: its source line, then the text.

    The source names the document's revision when it is known ("AC 43.13-1B
    CHG 1"), else its title, then the page and that page's issue date.

    The figure check reads the same text (`for_verification=True`), so a page
    the answer copies from the source line ("page 7-9 (PDF page 364)") counts as
    supported; without it every page citation was flagged as an unsupported
    figure. The issue date is left out of that text: "2001-09-27" would make a
    wrong "27" or "9" in the answer look supported.
    """
    meta = chunk.metadata_ or {}
    source = meta.get("revision") or meta.get("source", "unknown")
    where = _where(chunk, issued=not for_verification)
    return f"Source: {source}{where}\n{chunk.content}"


def build_context(chunks: list[tuple]) -> str:
    """The numbered CONTEXT blocks alone; block number == citation number."""
    blocks = [
        f"[{rank}] {context_block(chunk)}"
        for rank, (chunk, _distance) in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, chunks: list[tuple]) -> str:
    """Numbered CONTEXT blocks, then the question."""
    context = build_context(chunks)
    return f"CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {question}\n\nANSWER:"


async def prepare_answer(
    session: AsyncSession,
    ollama: OllamaClient,
    question: str,
    top_k: int | None = None,
) -> PreparedAnswer:
    """Retrieve and gate by distance ceiling; build the prompt if answerable.

    Retrieval goes through the Phase 1 pipeline (hybrid search + reranking,
    each independently feature-flagged in settings; both default off, so
    behavior is unchanged from Phase 0 until they are turned on).

    The DB session is committed (its pooled connection released) before this
    returns, so callers can spend seconds in the LLM -- or stream for minutes --
    without holding a connection.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k

    # With the answer cache on, the question is embedded here, once: retrieval
    # uses the vector, and the cache compares it with earlier questions'.
    query_embedding: list[float] | None = None
    if settings.answer_cache_enabled:
        (query_embedding,) = await ollama.embed([question])

    retrieved = await retrieve(
        session, ollama, question, top_k, query_embedding=query_embedding
    )

    relevant = select_context(
        retrieved, settings.max_distance, settings.context_max_distance
    )
    # The word lookup is cheap, so it also runs for a question the distance gate
    # refused: the reply can then say *which* words no document contains. It
    # only changes the outcome ("refuse") for a question the gate let through.
    unknown_terms: list[str] = []
    if settings.answerability_mode != "off":
        found = await check_answerability(
            session, question, settings.answerability_min_chunks
        )
        unknown_terms = found.unknown
        if (
            relevant
            and settings.answerability_mode == "refuse"
            and len(unknown_terms) >= settings.answerability_min_unknown
        ):
            await session.commit()
            return PreparedAnswer(
                question=question,
                refused=True,
                chunks=retrieved,
                refusal_reason=REASON_UNKNOWN_TERMS,
                unknown_terms=unknown_terms,
            )

    # The gates above judge the retrieved hits only; neighbours are added after
    # them, so widening the context can never turn a refusal into an answer.
    if relevant and settings.context_neighbors > 0:
        relevant = await expand_with_neighbours(
            session, relevant, settings.context_neighbors, settings.context_max_chunks
        )

    # Safety labels in exactly what the model will read (plus the start of each
    # passage's paragraph), found by code so the model cannot leave them out.
    # Before the commit: the paragraph lookback reads neighbouring chunks.
    warnings: list[SafetyWarning] = []
    sources: list[SourceRevision] = []
    if relevant:
        warnings = await find_warnings(
            session, relevant, settings.warning_lookback_chunks
        )
        # Which revision each passage's document is, and whether a newer one
        # is stored: every answer names its revision.
        sources = await find_source_revisions(session, relevant)

    await session.commit()

    if not relevant:
        return PreparedAnswer(
            question=question,
            refused=True,
            chunks=retrieved,
            refusal_reason=REASON_DISTANCE,
            unknown_terms=unknown_terms,
        )

    # The answer check runs last, on exactly the passages the answer would get,
    # and only for questions every cheaper gate let through.
    answer_check: str | None = None
    if settings.answer_check != "off":
        answer_check = await check_answerable(ollama, question, relevant)
        if settings.answer_check == "refuse" and answer_check == CHECK_NO:
            return PreparedAnswer(
                question=question,
                refused=True,
                chunks=relevant,
                refusal_reason=REASON_NOT_IN_CONTEXT,
                unknown_terms=unknown_terms,
                answer_check=answer_check,
            )

    active = system_prompt()
    prompt = build_prompt(question, relevant)
    warn_if_prompt_may_not_fit(active.text, prompt)
    return PreparedAnswer(
        question=question,
        refused=False,
        chunks=relevant,
        system=active.text,
        prompt=prompt,
        prompt_version=active.version,
        unknown_terms=unknown_terms,
        answer_check=answer_check,
        query_embedding=query_embedding,
        warnings=warnings,
        sources=sources,
    )


_QUESTION_SLOT = "\x00question\x00"


def cache_bucket(prepared: PreparedAnswer, llm: OllamaClient) -> str:
    """Hash of everything besides the question that shapes the answer.

    Two questions share cached answers only within one bucket: the same
    passages in the same order (so every [n] points at the same text), the
    same system prompt, the same answering model and generation settings,
    and the same embedding model (vectors from different models are not
    comparable). A change to any of them is a new bucket, never a stale hit.

    The prompt goes in whole, with the question left as a placeholder, so a
    change to its layout (the source line gaining the revision and issue, the
    wrapper around the question) also starts a new bucket: an answer written
    before the model was shown the revision is never served after.
    """
    settings = get_settings()
    material = {
        "embedding_model": settings.embedding_model,
        "client": type(llm).__name__,
        "model": getattr(llm, "model_name", None),
        "temperature": settings.temperature,
        "num_ctx": settings.llm_num_ctx,
        "max_tokens": settings.anthropic_max_tokens,
        "system": prepared.system,
        "prompt": build_prompt(_QUESTION_SLOT, prepared.chunks),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def lookup_cached_answer(
    cache: AnswerCache | None, llm: OllamaClient, prepared: PreparedAnswer
) -> CacheHit | None:
    """A stored answer for this exact context and an equivalent question."""
    if cache is None or prepared.refused or prepared.query_embedding is None:
        return None
    hit = await cache.lookup(
        cache_bucket(prepared, llm), prepared.question, prepared.query_embedding
    )
    if hit is not None:
        logger.info(
            "answer cache hit",
            similarity=round(hit.similarity, 4),
            age_seconds=round(hit.age_seconds),
        )
    return hit


async def remember_answer(
    cache: AnswerCache | None,
    llm: OllamaClient,
    prepared: PreparedAnswer,
    answer: str,
    verification: Verification | None,
) -> None:
    """Store a freshly written answer, unless the figure check flagged it.

    A flagged answer is still returned to its own caller, but it is not
    handed to anyone else: the next caller gets a new attempt instead.
    """
    if cache is None or prepared.refused or prepared.query_embedding is None:
        return
    if not answer.strip() or (verification is not None and not verification.ok):
        return
    await cache.store(
        cache_bucket(prepared, llm),
        prepared.question,
        prepared.query_embedding,
        answer,
        prepared.prompt_version,
    )


def verify_prepared(prepared: PreparedAnswer, answer: str) -> Verification | None:
    """Check `answer` against the passages that were in its prompt (None if off)."""
    if get_settings().verify_mode == "off":
        return None
    passages = [
        context_block(chunk, for_verification=True) for chunk, _ in prepared.chunks
    ]
    return verify_answer(answer, passages, prepared.question)


async def answer_question(
    session: AsyncSession,
    ollama: OllamaClient,
    question: str,
    top_k: int | None = None,
    cache: AnswerCache | None = None,
) -> AnswerResult:
    """Retrieve, gate by distance ceiling, then generate — or refuse.

    The refusal is structural: if nothing clears MAX_DISTANCE, the LLM
    is never called. `cache`: the semantic answer cache; it is consulted only
    after every gate, so it can replace the answer-writing call, never a gate.
    """
    prepared = await prepare_answer(session, ollama, question, top_k)
    if prepared.refused:
        return AnswerResult(
            question=question,
            refused=True,
            chunks=prepared.chunks,
            refusal_reason=prepared.refusal_reason,
            unknown_terms=prepared.unknown_terms,
            answer_check=prepared.answer_check,
        )

    assert prepared.system is not None and prepared.prompt is not None
    hit = await lookup_cached_answer(cache, ollama, prepared)
    if hit is not None:
        answer = hit.answer
    else:
        answer = await ollama.chat(prepared.system, prepared.prompt)
    # Re-checked on a hit too: the figure check reads the question as well.
    verification = verify_prepared(prepared, answer)
    if hit is None:
        await remember_answer(cache, ollama, prepared, answer, verification)
    warnings = mark_cited(prepared.warnings, parse_citations(answer))
    if (
        verification is not None
        and not verification.ok
        and get_settings().verify_mode == "block"
    ):
        logger.warning(
            "answer withheld: figures not in cited passages",
            unsupported=verification.unsupported,
            unit_mismatch=verification.unit_mismatch,
        )
        return AnswerResult(
            question=question,
            refused=True,
            chunks=prepared.chunks,
            prompt_version=prepared.prompt_version,
            refusal_reason=REASON_UNVERIFIED,
            unknown_terms=prepared.unknown_terms,
            verification=verification,
            answer_check=prepared.answer_check,
            cached=hit is not None,
            # The reader is sent to the passages themselves: their safety text
            # and revision go with them.
            warnings=warnings,
            sources=prepared.sources,
        )
    return AnswerResult(
        question=question,
        refused=False,
        answer=answer,
        chunks=prepared.chunks,
        prompt_version=prepared.prompt_version,
        unknown_terms=prepared.unknown_terms,
        verification=verification,
        answer_check=prepared.answer_check,
        cached=hit is not None,
        warnings=warnings,
        sources=prepared.sources,
    )


def stream_answer(
    ollama: OllamaClient, prepared: PreparedAnswer
) -> AsyncGenerator[str, None]:
    """Token stream for an answerable query. Refusals can never reach the model."""
    if prepared.refused or prepared.system is None or prepared.prompt is None:
        raise ValueError("cannot stream an answer for a refused query")
    return ollama.chat_stream(prepared.system, prepared.prompt)
