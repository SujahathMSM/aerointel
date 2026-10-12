from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.llm.ollama import OllamaClient
from app.prompts import Prompt, get_prompt
from app.services.pipeline import retrieve

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


def build_prompt(question: str, chunks: list[tuple]) -> str:
    """Numbered CONTEXT blocks; block number == citation number."""
    blocks = []
    for rank, (chunk, _distance) in enumerate(chunks, start=1):
        source = chunk.metadata_.get("source", "unknown")
        blocks.append(f"[{rank}] Source: {source}\n{chunk.content}")

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

    await session.commit()
    relevant = [
        (chunk, distance)
        for chunk, distance in retrieved
        if distance <= settings.max_distance
    ]

    if not relevant:
        return PreparedAnswer(question=question, refused=True, chunks=retrieved)

    active = system_prompt()
    return PreparedAnswer(
        question=question,
        refused=False,
        chunks=relevant,
        system=active.text,
        prompt=build_prompt(question, relevant),
        prompt_version=active.version,
    )


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
        return AnswerResult(question=question, refused=True, chunks=prepared.chunks)

    assert prepared.system is not None and prepared.prompt is not None
    answer = await ollama.chat(prepared.system, prepared.prompt)
    return AnswerResult(
        question=question,
        refused=False,
        answer=answer,
        chunks=prepared.chunks,
        prompt_version=prepared.prompt_version,
    )


def stream_answer(
    ollama: OllamaClient, prepared: PreparedAnswer
) -> AsyncGenerator[str, None]:
    """Token stream for an answerable query. Refusals can never reach the model."""
    if prepared.refused or prepared.system is None or prepared.prompt is None:
        raise ValueError("cannot stream an answer for a refused query")
    return ollama.chat_stream(prepared.system, prepared.prompt)
