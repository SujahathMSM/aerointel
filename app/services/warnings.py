"""Safety text the answer must never drop: WARNING, CAUTION and NOTE labels.

The manual puts its safety text under labels ("WARNING: Do not probe cuts or
embedded foreign objects while tire is inflated."). A model asked about tire
cuts may answer correctly and leave that sentence out, which is the dangerous
kind of correct answer. So the labels are found by code, in the passages the
model reads, and returned beside the answer word for word -- the model does
not decide whether they are shown. They never refuse or block an answer.

Where a warning ends, in running page text with no line breaks: at the first
list item (" c. Tread Cuts"), figure or table caption, SECTION heading,
numbered paragraph heading ("6-181. TITANIUM"), or the next label. A label
without a colon (a table banner: "CAUTION THE FOLLOWING TORQUE VALUES ARE
...") keeps one sentence. Anything longer than MAX_WARNING_CHARS is cut at the
last sentence end before it.

A chunk is cut on sentence ends, so a warning can sit in the chunk before the
one retrieved (at the top of its paragraph) or run on into the next one. With
the chunks' positions known, each read chunk is joined to its neighbours
(overlap removed) and warnings are read from that window: one that starts in
the read chunk is completed from the next chunk if it runs on, and one at the
start of the same numbered paragraph, at most `lookback` chunks back, is
returned with source "paragraph_start" and its page (it is not in a block the
model read, so it gets no [n]).
"""

import re
from dataclasses import dataclass, replace

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.services.pdf_text import _HEADING, _SENTENCE_END, _starts_a_paragraph

LEVELS = ("WARNING", "CAUTION", "NOTE")
MAX_WARNING_CHARS = 600
# Chunks read past the end of a read chunk to finish a warning that runs on.
LOOKAHEAD_CHUNKS = 2
# A warning body shorter than this is a reference ("see NOTE"), not a warning.
MIN_WARNING_CHARS = 3
# Consecutive chunks repeat up to CHUNK_OVERLAP_CHARS of text; a shared run
# shorter than this is a coincidence, not overlap.
MIN_OVERLAP_CHARS = 20
MAX_OVERLAP_CHARS = 600

SOURCE_BLOCK = "block"
SOURCE_PARAGRAPH_START = "paragraph_start"

# Upper case only: "the note in the manufacturer's manual" is not a label.
_LABEL = re.compile(r"\b(WARNING|CAUTION|NOTE)S?\b(:?)\s*")
_LIST_ITEM = re.compile(r"\s(?:[a-z]\.|\(\d{1,2}\)|\([a-z]\))\s+(?=[A-Z])")
_CAPTION = re.compile(r"\s(?:FIGURE|FIG\.|TABLE)\s+\d")
_SECTION = re.compile(r"\sSECTION\s+\d")


@dataclass(frozen=True)
class FoundWarning:
    """A label and its text, located in one piece of running text."""

    level: str
    text: str
    start: int  # offset of the label
    end: int  # offset just past the text


@dataclass(frozen=True)
class SafetyWarning:
    """A warning returned with an answer."""

    level: str
    text: str  # verbatim, without the label
    # [n] of the block it was read in; None when found at the paragraph start.
    block: int | None
    source: str  # SOURCE_BLOCK | SOURCE_PARAGRAPH_START
    page: int | None
    page_label: str | None
    # The block (or, for a paragraph start, the block it leads into) is cited
    # by the answer. Unknown before the answer is written, so False until then.
    cited: bool = False
    # Internal, for `cited`: the blocks it is in; for a paragraph start (in no
    # read block), the blocks of that paragraph it leads into.
    for_blocks: tuple[int, ...] = ()


def _label_starts(text: str, start: int) -> bool:
    """A label starts a statement: after a sentence end or a capitalised title.

    "See NOTE 3" and "per the WARNING above" are references, not labels.
    """
    before = text[:start].rstrip()
    if not before:
        return True
    if before[-1] in ".!?:;)]\"'":
        return True
    word = before.split()[-1]
    return word.isupper() and any(ch.isalpha() for ch in word)


def _heading_positions(text: str) -> list[int]:
    return [
        m.start()
        for m in _HEADING.finditer(text)
        if _starts_a_paragraph(text, m.start())
    ]


def _first_end(text: str, body_start: int, colon: bool) -> int:
    """Where the warning whose text starts at `body_start` ends."""
    ends = [len(text)]
    for pattern in (_LIST_ITEM, _CAPTION, _SECTION):
        if m := pattern.search(text, body_start):
            ends.append(m.start())
    for m in _LABEL.finditer(text, body_start):
        if _label_starts(text, m.start()):
            ends.append(m.start())
            break
    ends.extend(h for h in _heading_positions(text) if h > body_start)
    end = min(ends)
    # A banner without a colon ("CAUTION THE FOLLOWING ...") keeps one sentence.
    if not colon and (m := _SENTENCE_END.search(text, body_start, end)):
        end = m.start()
    return end


def _cap(body: str) -> str:
    if len(body) <= MAX_WARNING_CHARS:
        return body
    head = body[:MAX_WARNING_CHARS]
    last = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if last > 0:
        return head[: last + 1]
    if head.endswith((".", "!", "?")):
        return head
    return head.rsplit(" ", 1)[0]


def extract_warnings(text: str) -> list[FoundWarning]:
    """Every labelled warning in `text`, in reading order."""
    found: list[FoundWarning] = []
    for m in _LABEL.finditer(text):
        if not _label_starts(text, m.start()):
            continue
        end = _first_end(text, m.end(), colon=bool(m.group(2)))
        body = _cap(text[m.end() : end].strip())
        if len(body) < MIN_WARNING_CHARS:
            continue
        found.append(FoundWarning(m.group(1), body, m.start(), m.start() + len(body)))
    return found


# ------------------------------------------------------------------ windows


@dataclass(frozen=True)
class Piece:
    """One chunk's text inside a window, with where it came from."""

    start: int
    end: int
    page: int | None
    page_label: str | None


def _overlap(a: str, b: str) -> int:
    """Length of the longest end of `a` that `b` starts with (0 if too short)."""
    for k in range(min(len(a), len(b), MAX_OVERLAP_CHARS), MIN_OVERLAP_CHARS - 1, -1):
        if a.endswith(b[:k]):
            return k
    return 0


def join_chunks(chunks: list) -> tuple[str, list[Piece]]:
    """Consecutive chunks as one text, the overlap each repeats removed.

    Each piece's span covers the chunk's own text in the joined string, so
    overlapping chunks have overlapping spans.
    """
    text = ""
    pieces: list[Piece] = []
    for chunk in chunks:
        content = chunk.content
        shared = _overlap(text, content) if text else 0
        if shared:
            start = len(text) - shared
            text += content[shared:]
        else:
            start = len(text) + (1 if text else 0)
            text = f"{text} {content}" if text else content
        pieces.append(
            Piece(
                start,
                start + len(content),
                getattr(chunk, "page", None),
                getattr(chunk, "page_label", None),
            )
        )
    return text, pieces


def _piece_at(pieces: list[Piece], offset: int) -> Piece:
    for piece in pieces:
        if piece.start <= offset < piece.end:
            return piece
    return pieces[-1]


def warnings_in_window(
    window: list, read_index: int, block: int
) -> list[SafetyWarning]:
    """Warnings for the read chunk `window[read_index]`, numbered `block`.

    Those starting in the read chunk (completed from later chunks if they run
    on), and those earlier in the same numbered paragraph.
    """
    text, pieces = join_chunks(window)
    read = pieces[read_index]
    headings = [h for h in _heading_positions(text) if h <= read.start]
    paragraph_start = headings[-1] if headings else 0
    out: list[SafetyWarning] = []
    for w in extract_warnings(text):
        if read.start <= w.start < read.end:
            source, number = SOURCE_BLOCK, block
        elif paragraph_start <= w.start < read.start:
            source, number = SOURCE_PARAGRAPH_START, None
        else:
            continue
        where = _piece_at(pieces, w.start)
        out.append(
            SafetyWarning(
                level=w.level,
                text=w.text,
                block=number,
                source=source,
                page=where.page,
                page_label=where.page_label,
                for_blocks=(block,),
            )
        )
    return out


def _position(chunk) -> tuple[int, int] | None:
    document_id = getattr(chunk, "document_id", None)
    index = getattr(chunk, "chunk_index", None)
    if document_id is None or index is None:
        return None
    return document_id, index


def _window(chunk, by_position: dict, lookback: int) -> tuple[list, int]:
    """The read chunk with its stored neighbours: (chunks in order, its index)."""
    pos = _position(chunk)
    if pos is None:
        return [chunk], 0
    document_id, index = pos
    before: list = []
    for i in range(index - 1, index - 1 - lookback, -1):
        neighbour = by_position.get((document_id, i))
        if neighbour is None:
            break
        before.insert(0, neighbour)
    after: list = []
    for i in range(index + 1, index + 1 + LOOKAHEAD_CHUNKS):
        neighbour = by_position.get((document_id, i))
        if neighbour is None:
            break
        after.append(neighbour)
    return [*before, chunk, *after], len(before)


def merge(found: list[SafetyWarning]) -> list[SafetyWarning]:
    """Drop repeats (overlapping chunks, windows) and sort most severe first.

    The same warning seen twice stays where it was first seen, but a sighting
    inside a read block beats one at a paragraph start (it has a real [n]), and
    a cut-off copy gives way to the complete text (one starts with the other).
    A warning inside read blocks belongs to those blocks only: that a later
    block of the same paragraph also found it by lookback must not make it
    "cited" when the answer cites only that later block (seen on the real
    manual: a NOTE in [3] marked cited by an answer citing [1]). The sort is
    stable: within a level, cited before uncited, then reading order.
    """
    kept: list[SafetyWarning] = []
    for w in found:
        dup = next(
            (
                i
                for i, k in enumerate(kept)
                if k.level == w.level
                and (k.text.startswith(w.text) or w.text.startswith(k.text))
            ),
            None,
        )
        if dup is None:
            kept.append(w)
            continue
        k = kept[dup]
        best = w if k.source != SOURCE_BLOCK and w.source == SOURCE_BLOCK else k
        # Only sightings of the same kind as the one kept count.
        same = [x for x in (k, w) if x.source == best.source]
        kept[dup] = replace(
            best,
            text=max(k.text, w.text, key=len),
            cited=any(x.cited for x in same),
            for_blocks=tuple(dict.fromkeys(b for x in same for b in x.for_blocks)),
        )
    return sorted(kept, key=lambda w: (LEVELS.index(w.level), not w.cited))


def warnings_for(
    chunks: list[tuple], neighbours: dict | None = None, lookback: int = 4
) -> list[SafetyWarning]:
    """Warnings for the passages the model reads, in prompt order ([n] = i + 1).

    `neighbours`: {(document_id, chunk_index): chunk} of stored chunks around
    them (see `fetch_neighbours`); without it each chunk is read on its own.
    """
    by_position = dict(neighbours or {})
    for chunk, _ in chunks:
        if pos := _position(chunk):
            by_position.setdefault(pos, chunk)
    found: list[SafetyWarning] = []
    for number, (chunk, _distance) in enumerate(chunks, start=1):
        window, read_index = _window(chunk, by_position, lookback)
        found.extend(warnings_in_window(window, read_index, number))
    return merge(found)


def mark_cited(warnings: list[SafetyWarning], cited: list[int]) -> list[SafetyWarning]:
    """Flag warnings whose block the answer cites, and re-sort (cited first)."""
    blocks = set(cited)
    return merge([replace(w, cited=bool(blocks & set(w.for_blocks))) for w in warnings])


async def fetch_neighbours(
    session: AsyncSession, chunks: list[tuple], lookback: int
) -> dict[tuple[int, int], DocumentChunk]:
    """The stored chunks around each read chunk: `lookback` before, a few after."""
    keys = set()
    for chunk, _ in chunks:
        if (pos := _position(chunk)) is None:
            continue
        document_id, index = pos
        for i in range(index - lookback, index + LOOKAHEAD_CHUNKS + 1):
            if i >= 0 and i != index:
                keys.add((document_id, i))
    if not keys:
        return {}
    rows = (
        (
            await session.execute(
                select(DocumentChunk).where(
                    tuple_(DocumentChunk.document_id, DocumentChunk.chunk_index).in_(
                        list(keys)
                    )
                )
            )
        )
        .scalars()
        .all()
    )
    return {
        (row.document_id, row.chunk_index): row
        for row in rows
        if row.document_id is not None and row.chunk_index is not None
    }


async def find_warnings(
    session: AsyncSession, chunks: list[tuple], lookback: int
) -> list[SafetyWarning]:
    """Warnings for the passages the model reads (one DB query for neighbours)."""
    neighbours = await fetch_neighbours(session, chunks, lookback) if chunks else {}
    return warnings_for(chunks, neighbours, lookback)
