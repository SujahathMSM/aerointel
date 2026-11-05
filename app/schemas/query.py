from datetime import date

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    id: int
    distance: float
    source: str | None = None
    content: str
    # Where the passage sits when it came from an uploaded PDF (else None):
    # the document, the PDF page number, and the page number printed on the page.
    document_id: int | None = None
    page: int | None = None
    page_label: str | None = None
    # Which issue of the document the passage is from ("AC 43.13-1B CHG 1"),
    # and the issue date printed on its own page (a change-page document
    # mixes issues). None for documents ingested before revisions were stored.
    revision: str | None = None
    issued: str | None = None


class SourceOut(BaseModel):
    """A document the answer reads from, and how current its revision is."""

    document_id: int
    title: str
    revision: str | None = None
    revision_date: date | None = None
    stored_on: date | None = None
    # Marked by hand as replaced: do not use without checking.
    superseded: bool = False
    # A newer revision of the same document is stored (but was not read).
    newer_revision: str | None = None
    newer_document_id: int | None = None
    # Plain-language summary: which revision, stored when, confirm it is current.
    note: str


class WarningOut(BaseModel):
    """WARNING / CAUTION / NOTE text from a passage, copied word for word."""

    level: str  # "WARNING", "CAUTION" or "NOTE", most severe first
    text: str  # verbatim, without the label
    # [n] of the context block it is in; None when it was found at the start
    # of the paragraph that block belongs to (source "paragraph_start").
    block: int | None = None
    source: str  # "block" | "paragraph_start"
    page: int | None = None
    page_label: str | None = None
    # The answer cites its block (or the block its paragraph leads into).
    # Always false on the stream's `meta`, sent before the answer exists.
    cited: bool = False


class VerificationOut(BaseModel):
    """Numbers in the answer checked against the passages it cites."""

    ok: bool
    checked: int  # numbers found in the answer
    cited: list[int] = []  # context blocks the answer cites
    unsupported: list[str] = []  # figures no cited passage contains
    unit_mismatch: list[str] = []  # figure present, but with another unit
    bad_citations: list[int] = []  # [n] pointing at a block that does not exist
    # unsupported figure -> uncited blocks that contain it (likely a wrong citation)
    found_in_uncited: dict[str, list[int]] = {}
    blocked: bool = False  # the answer was withheld because of the above


class QueryResponse(BaseModel):
    question: str
    refused: bool
    answer: str | None = None
    refusal_message: str | None = None
    chunks: list[RetrievedChunk] = []
    # Which prompt version produced the answer; None on a refusal, where no
    # prompt was used because the model was never called.
    prompt_version: str | None = None
    # Which model wrote the answer (e.g. "qwen3:4b-instruct", "claude-haiku-5-5");
    # None on a refusal, where no answer was written.
    model: str | None = None
    # Why a refusal happened: "too_far" (nothing close enough), "unknown_terms"
    # (a word in the question appears in no stored chunk), "not_in_context" (the
    # answer check judged that the passages do not answer it) or
    # "unverified_numbers" (the answer was withheld because its figures are not
    # in the cited passages).
    refusal_reason: str | None = None
    # Question words found in no stored chunk. Reported on answers too: an answer
    # to a question with an unknown word is probably about something else.
    unknown_terms: list[str] = []
    verification: VerificationOut | None = None
    # The model's own judgement, before answering, of whether the passages
    # answer the question: "yes", "no" or "unclear". None when ANSWER_CHECK=off.
    answer_check: str | None = None
    # Safety labels from the passages the model read, never dropped and never
    # chosen by the model. Empty on a refusal.
    warnings: list[WarningOut] = []
    # The documents the answer reads from: revision and whether it is current.
    # Empty on a refusal.
    sources: list[SourceOut] = []
    # The answer was reused from the semantic cache: written earlier for an
    # equivalent question over these exact passages, not by a new model call.
    cached: bool = False
