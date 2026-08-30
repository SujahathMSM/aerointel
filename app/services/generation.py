from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.llm.ollama import OllamaClient
from app.services.retrieval import search_chunks

SYSTEM_PROMPT = """You are AeroIntel, an aviation technical assistant.

You answer strictly from the numbered CONTEXT blocks given with each \
question. Do not use outside knowledge. Do not fill gaps with what \
aircraft systems normally do.

Rules:
1. Use only the CONTEXT. Cite the block you used, like [1] or [2, 3].
2. If the CONTEXT does not answer the question, say so plainly. Do not \
guess.
3. Quote exact figures and limits from the CONTEXT rather than rounding \
or paraphrasing them.
4. Be concise. No preamble."""

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

    The refusal is structural: if nothing clears MAX_DISTANCE, the LLM
    is never called.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k

    retrieved = await search_chunks(session, ollama, question, top_k)
    relevant = [
        (chunk, distance)
        for chunk, distance in retrieved
        if distance <= settings.max_distance
    ]

    if not relevant:
        return AnswerResult(question=question, refused=True, chunks=retrieved)

    prompt = build_prompt(question, relevant)
    answer = await ollama.chat(SYSTEM_PROMPT, prompt)
    return AnswerResult(
        question=question, refused=False, answer=answer, chunks=relevant
    )
