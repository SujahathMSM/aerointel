"""Reranking: rescore retrieved candidates by (query, text) relevance.

A bi-encoder (EmbeddingGemma) compares query and document *separately* —
fast, but it can't see how the words interact. A cross-encoder reads the
query and the document *together* and scores the pair directly: slower
(one forward pass per pair), but far more precise about which of the top
candidates actually answers the question. That is why reranking only
touches the top-N candidates, never the whole corpus.
"""

from typing import Protocol

from app.core.config import get_settings


class Reranker(Protocol):
    """Anything that can score (query, document) pairs by relevance."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, same order as input."""
        ...


class NoReranker:
    """Feature-flag off: keep the retrieval order untouched."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return list(range(len(documents), 0, -1))


class CrossEncoderReranker:
    """Cross-encoder via sentence-transformers, loaded on first use.

    Lazy import + lazy model load: importing this module or constructing
    the object must not require torch — only calling score() does. That
    keeps the app startable (and the Docker image lean) while the
    reranker is disabled.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "reranker='crossencoder' requires the optional "
                    "dependency. Install it with: "
                    "pip install -r requirements-rerank.txt"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        model = self._load()
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in model.predict(pairs)]


def build_reranker(name: str, model_name: str) -> Reranker:
    """Pure factory: name + model -> reranker. Unknown names fail loudly."""
    if name == "none":
        return NoReranker()
    if name == "crossencoder":
        return CrossEncoderReranker(model_name)
    raise ValueError(f"unknown reranker {name!r} (expected 'none' or 'crossencoder')")


def get_reranker() -> Reranker:
    """Build the reranker configured in settings."""
    settings = get_settings()
    return build_reranker(settings.reranker, settings.reranker_model)


def rerank(
    query: str,
    results: list[tuple],
    reranker: Reranker,
    top_k: int,
) -> list[tuple]:
    """Rescore candidates, return the best top_k in new relevance order.

    results: (chunk, distance) pairs from the retrieval stage. Distances
    are carried along untouched — the generation distance gate still
    judges them; the reranker only decides *ordering*.
    """
    if not results:
        return []
    scores = reranker.score(query, [chunk.content for chunk, _ in results])
    ordered = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
    return [result for result, _ in ordered[:top_k]]
