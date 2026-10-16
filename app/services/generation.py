from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.ollama import OllamaClient
from app.prompts import Prompt, get_prompt
from app.services.answerability import check_answerability
from app.services.context import expand_with_neighbours
from app.services.pipeline import retrieve
from app.services.verification import Verification, verify_answer

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

REASON_DISTANCE = "too_far"
REASON_UNKNOWN_TERMS = "unknown_terms"
REASON_UNVERIFIED = "unverified_numbers"


def refusal_message(
    reason: str | None, unknown_terms: list[str], figures: list[str], ceiling: float
) -> str:
    """The text that goes with a refusal, chosen by why it was refused."""
    if reason == REASON_UNKNOWN_TERMS:
        return UNKNOWN_TERMS_MESSAGE.format(terms=", ".join(unknown_terms))
    if reason == REASON_UNVERIFIED:
        return UNVERIFIED_MESSAGE.format(figures=", ".join(figures))
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


def _where(chunk) -> str:
    """', page 3-3 (PDF page 120)' for chunks cut from a PDF; '' otherwise."""
    page = getattr(chunk, "page", None)
    if page is None:
        return ""
    label = getattr(chunk, "page_label", None)
    return f", page {label} (PDF page {page})" if label else f", PDF page {page}"


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


def build_prompt(question: str, chunks: list[tuple]) -> str:
    """Numbered CONTEXT blocks; block number == citation number."""
    blocks = []
    for rank, (chunk, _distance) in enumerate(chunks, start=1):
        source = chunk.metadata_.get("source", "unknown")
        blocks.append(f"[{rank}] Source: {source}{_where(chunk)}\n{chunk.content}")

    context = "\n\n---\n\n".join(blocks)
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

    retrieved = await retrieve(session, ollama, question, top_k)

    relevant = [
        (chunk, distance)
        for chunk, distance in retrieved
        if distance <= settings.max_distance
    ]
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

    await session.commit()

    if not relevant:
        return PreparedAnswer(
            question=question,
            refused=True,
            chunks=retrieved,
            refusal_reason=REASON_DISTANCE,
            unknown_terms=unknown_terms,
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
    )


def verify_prepared(prepared: PreparedAnswer, answer: str) -> Verification | None:
    """Check `answer` against the passages that were in its prompt (None if off)."""
    if get_settings().verify_mode == "off":
        return None
    passages = [chunk.content for chunk, _distance in prepared.chunks]
    return verify_answer(answer, passages, prepared.question)


async def answer_question(
    session: AsyncSession,
    ollama: OllamaClient,
    question: str,
    top_k: int | None = None,
) -> AnswerResult:
    """Retrieve, gate by distance ceiling, then generate — or refuse.

    The refusal is structural: if nothing clears MAX_DISTANCE, the LLM
    is never called.
    """
    prepared = await prepare_answer(session, ollama, question, top_k)
    if prepared.refused:
        return AnswerResult(
            question=question,
            refused=True,
            chunks=prepared.chunks,
            refusal_reason=prepared.refusal_reason,
            unknown_terms=prepared.unknown_terms,
        )

    assert prepared.system is not None and prepared.prompt is not None
    answer = await ollama.chat(prepared.system, prepared.prompt)
    verification = verify_prepared(prepared, answer)
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
        )
    return AnswerResult(
        question=question,
        refused=False,
        answer=answer,
        chunks=prepared.chunks,
        prompt_version=prepared.prompt_version,
        unknown_terms=prepared.unknown_terms,
        verification=verification,
    )


def stream_answer(
    ollama: OllamaClient, prepared: PreparedAnswer
) -> AsyncGenerator[str, None]:
    """Token stream for an answerable query. Refusals can never reach the model."""
    if prepared.refused or prepared.system is None or prepared.prompt is None:
        raise ValueError("cannot stream an answer for a refused query")
    return ollama.chat_stream(prepared.system, prepared.prompt)
