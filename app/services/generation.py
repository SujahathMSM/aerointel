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


def build_prompt(question: str, chunks: list[tuple]) -> str:
    """Numbered CONTEXT blocks; block number == citation number."""
    blocks = []
    for rank, (chunk, _distance) in enumerate(chunks, start=1):
        source = chunk.metadata_.get("source", "unknown")
        blocks.append(f"[{rank}] Source: {source}\n{chunk.content}")

    context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {question}\n\nANSWER:"


async def answer_question(
    session: AsyncSession,
    ollama: OllamaClient,
    question: str,
    top_k: int | None = None,
) -> AnswerResult:
    """Retrieve, gate by distance ceiling, then generate — or refuse.

    Retrieval goes through the Phase 1 pipeline (hybrid search + reranking,
    each independently feature-flagged in settings; both default off, so
    behavior is unchanged from Phase 0 until they are turned on).

    The refusal is structural: if nothing clears MAX_DISTANCE, the LLM
    is never called.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k

    retrieved = await retrieve(session, ollama, question, top_k)
    relevant = [
        (chunk, distance)
        for chunk, distance in retrieved
        if distance <= settings.max_distance
    ]

    if not relevant:
        return AnswerResult(question=question, refused=True, chunks=retrieved)

    active = system_prompt()
    prompt = build_prompt(question, relevant)
    answer = await ollama.chat(active.text, prompt)
    return AnswerResult(
        question=question,
        refused=False,
        answer=answer,
        chunks=relevant,
        prompt_version=active.version,
    )
