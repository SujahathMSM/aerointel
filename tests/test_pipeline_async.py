"""The cross-encoder is synchronous CPU work. retrieve() must run it off the
event loop, or every other request freezes while it scores."""

import asyncio
import time

from app.services import pipeline


class _Chunk:
    def __init__(self, cid: int, content: str):
        self.id = cid
        self.content = content
        self.metadata_ = {}


class _BlockingReranker:
    """Not a NoReranker, so retrieve() treats reranking as active."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        time.sleep(0.3)  # simulates model.predict() hogging a thread
        return [float(i) for i in range(len(documents))]


async def test_reranking_does_not_block_the_event_loop(monkeypatch):
    async def fake_search(session, ollama, query, top_k=5):
        return [(_Chunk(1, "a"), 0.1), (_Chunk(2, "b"), 0.2)]

    monkeypatch.setattr(pipeline, "search_chunks", fake_search)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        results = await pipeline.retrieve(
            None, None, "q", top_k=2, hybrid=False, reranker=_BlockingReranker()
        )
    finally:
        task.cancel()

    assert [chunk.id for chunk, _ in results] == [2, 1]  # reranked order
    # A blocked loop would manage ~0 ticks during the 0.3s sleep; a free
    # loop manages roughly 30. Generous threshold to stay stable on slow CI.
    assert ticks >= 10
