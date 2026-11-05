"""Which revision an answer comes from, and whether a newer one is stored.

Maintenance must be done to current data. Every document records its revision
("AC 43.13-1B CHG 1") and a key that stays the same across revisions ("AC
43.13-1B"). For each document an answer reads, the response says which
revision it is, when it was stored, and whether a newer revision of the same
document is stored or this one is marked superseded.

The system only knows what is stored. It cannot know whether the FAA has
published something newer since, so the note always asks the reader to confirm
the revision is current; there is no online check.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document


@dataclass(frozen=True)
class SourceRevision:
    """One document an answer reads from."""

    document_id: int
    title: str
    revision: str | None
    revision_date: date | None
    stored_on: date | None
    superseded: bool
    # The newest other stored revision of the same document, if it is newer.
    newer_revision: str | None = None
    newer_document_id: int | None = None
    newer_revision_date: date | None = None

    @property
    def note(self) -> str:
        name = self.revision or self.title
        if self.superseded:
            return (
                f"{name} is marked superseded. Do not use it for maintenance "
                "without checking the current revision."
            )
        if self.newer_revision:
            when = f" ({self.newer_revision_date})" if self.newer_revision_date else ""
            return (
                f"A newer revision is stored: {self.newer_revision}{when}. "
                f"This answer quotes {name}."
            )
        stored = f", as stored on {self.stored_on}" if self.stored_on else ""
        return f"From {name}{stored}; confirm it is the current revision before use."


def _newer(document: Document, candidate: Document) -> bool:
    """Is `candidate` a later revision of the same document than `document`?"""
    if candidate.id == document.id or candidate.document_key != document.document_key:
        return False
    if candidate.revision_date and document.revision_date:
        return candidate.revision_date > document.revision_date
    # Without both dates, a later upload (higher id) of a different revision
    # counts as newer.
    return candidate.revision != document.revision and candidate.id > document.id


def source_revisions(
    read: list[Document], same_key: list[Document]
) -> list[SourceRevision]:
    """One entry per document read, in first-read order (pure, so unit-tested)."""
    out = []
    for document in read:
        newer = [d for d in same_key if document.document_key and _newer(document, d)]
        newest = max(
            newer,
            key=lambda d: (d.revision_date or date.min, d.id),
            default=None,
        )
        out.append(
            SourceRevision(
                document_id=document.id,
                title=document.title,
                revision=document.revision,
                revision_date=document.revision_date,
                stored_on=document.created_at.date() if document.created_at else None,
                superseded=bool(document.superseded),
                newer_revision=newest.revision if newest else None,
                newer_document_id=newest.id if newest else None,
                newer_revision_date=newest.revision_date if newest else None,
            )
        )
    return out


async def find_source_revisions(
    session: AsyncSession, chunks: list[tuple]
) -> list[SourceRevision]:
    """Revision notes for the documents behind `chunks` (one or two queries)."""
    ids = list(
        dict.fromkeys(
            d for chunk, _ in chunks if (d := getattr(chunk, "document_id", None))
        )
    )
    if not ids:
        return []
    rows = (
        (await session.execute(select(Document).where(Document.id.in_(ids))))
        .scalars()
        .all()
    )
    by_id = {d.id: d for d in rows}
    read = [by_id[i] for i in ids if i in by_id]
    keys = {d.document_key for d in read if d.document_key}
    same_key: list[Document] = []
    if keys:
        same_key = list(
            (
                await session.execute(
                    select(Document).where(Document.document_key.in_(keys))
                )
            )
            .scalars()
            .all()
        )
    return source_revisions(read, same_key)
