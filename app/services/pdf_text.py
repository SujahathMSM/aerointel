"""PDF -> clean page text -> page-bound chunks.

Pure functions: no database, no network, no Ollama. That keeps the fiddly part
of document ingestion (what counts as a header, where a chunk ends) testable
with plain strings.

Three steps:

1. `extract_pages` reads the PDF's text layer page by page and, for pages that
   have none, optionally falls back to OCR.
2. Cleaning removes what is not content: running headers and footers that
   repeat on every page ("9/27/01  AC 43.13-1B  Par 3-2  Page 3-3"), and the
   line breaks and hyphenation that come from the page layout.
3. `chunk_pages` cuts each page into chunks that end on sentence boundaries.
   A chunk never crosses a page, so every citation points at one page. With
   `paragraphs=True` a chunk also starts at each numbered paragraph heading
   ("6-181. TITANIUM AND TITANIUM ALLOYS."), so one chunk does not carry the
   end of one topic and the start of the next.
"""

import io
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# A page with fewer letters and digits than this has no usable text layer.
MIN_PAGE_ALNUM = 30
# Chunks with fewer letters and digits than this are not worth embedding.
MIN_CHUNK_ALNUM = 40

# (pdf_bytes, page_numbers, language) -> {page_number: text}
OcrFn = Callable[[bytes, list[int], str], dict[int, str]]


class PdfError(ValueError):
    """The file cannot be turned into text (unreadable, encrypted, empty)."""


class PdfNoTextError(PdfError):
    """The PDF is readable but no page has any text (probably a scan)."""


@dataclass(frozen=True)
class PageText:
    number: int  # 1-based position in the PDF file
    label: str | None  # page number printed on the page, e.g. "3-3"
    text: str  # cleaned text (single spaces, no running header/footer)
    ocr: bool = False  # True if the text came from OCR


@dataclass(frozen=True)
class TextChunk:
    index: int  # 0-based position within the document
    page: int  # PDF page number
    page_label: str | None
    text: str


# --------------------------------------------------------------------- labels

# "Page 3-3", "Page xii", "Page 112" in a header or footer line.
_PAGE_WORD = re.compile(r"\bPage\s+([A-Za-z]{1,4}-?\d{0,4}|\d{1,4}(?:-\d{1,4})?)\b")
# A line that is nothing but a page number: "3-3", "112", "xii".
_BARE_NUMBER = re.compile(r"^(\d{1,4}(?:-\d{1,4})?|[ivxlc]{1,6})$", re.IGNORECASE)


def find_page_label(lines: list[str], edge: int = 3) -> str | None:
    """Read the printed page number from the first/last few lines of a page."""
    candidates = lines[:edge] + lines[-edge:]
    for line in candidates:
        match = _PAGE_WORD.search(line)
        if match:
            return match.group(1)
    for line in reversed(candidates):  # footers are far more common than headers
        stripped = line.strip()
        if _BARE_NUMBER.match(stripped):
            return stripped
    return None


# ------------------------------------------------------------------- cleaning


def _normalise_edge(line: str) -> str:
    """Make 'Page 3-3' and 'Page 3-4' compare equal: digits become '#'."""
    return " ".join(re.sub(r"\d+", "#", line.lower()).split())


def _edge_width(lines: list[str], edge: int) -> int:
    """How many lines at each end of a page count as 'edge'.

    A short page must keep its middle: with 3 lines only the first and last
    are edges, never all three, so a page that is mostly body text is not
    mistaken for pure header and footer.
    """
    return min(edge, len(lines) // 3)


def strip_repeated_edges(
    pages_lines: list[list[str]], edge: int = 3, min_share: float = 0.25
) -> list[list[str]]:
    """Drop first/last lines that repeat across many pages (headers, footers).

    A line counts as a running header/footer when, ignoring digits, it appears
    in the first or last `edge` lines of at least `min_share` of the pages
    (and at least 3 pages). Only edge lines are ever removed, so the same
    words in the body of a page are untouched.
    """
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        width = _edge_width(lines, edge)
        seen = {
            _normalise_edge(line)
            for line in lines[:width] + lines[len(lines) - width :]
            if line.strip()
        }
        counts.update(seen)

    threshold = max(3, min_share * len(pages_lines))
    repeated = {line for line, n in counts.items() if n >= threshold}

    cleaned = []
    for lines in pages_lines:
        width = _edge_width(lines, edge)
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < width or i >= len(lines) - width
            if at_edge and _normalise_edge(line) in repeated:
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned


def join_lines(lines: list[str]) -> str:
    """Turn layout lines into running text: undo hyphenation, single spaces."""
    text = "\n".join(line.strip() for line in lines if line.strip())
    # "inspec-\ntion" -> "inspection"; only when the next word is lower case,
    # so a real hyphen before a capital or number ("Part-\n145") survives.
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    return " ".join(text.split())


def _alnum_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


# ----------------------------------------------------------------- extraction


def extract_pages(
    data: bytes,
    ocr: OcrFn | None = None,
    ocr_language: str = "eng",
) -> list[PageText]:
    """Read every page's text, clean it, and OCR pages that have none."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("The PDF is password-protected.")
        raw_pages = []
        for page in reader.pages:
            try:
                raw_pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - one bad page must not sink a manual
                raw_pages.append("")
    except PdfError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError) as exc:
        raise PdfError(f"The file could not be read as a PDF: {exc}") from exc

    if not raw_pages:
        raise PdfError("The PDF has no pages.")

    lines_per_page = [
        [line for line in raw.splitlines() if line.strip()] for raw in raw_pages
    ]
    labels = [find_page_label(lines) for lines in lines_per_page]
    cleaned = strip_repeated_edges(lines_per_page)
    texts = [join_lines(lines) for lines in cleaned]

    blank = [
        i + 1 for i, text in enumerate(texts) if _alnum_count(text) < MIN_PAGE_ALNUM
    ]
    ocr_text: dict[int, str] = {}
    if blank and ocr is not None:
        ocr_text = ocr(data, blank, ocr_language)

    pages = []
    for i, text in enumerate(texts):
        number = i + 1
        used_ocr = False
        if number in ocr_text and _alnum_count(ocr_text[number]) >= MIN_PAGE_ALNUM:
            text = join_lines(ocr_text[number].splitlines())
            used_ocr = True
        pages.append(PageText(number=number, label=labels[i], text=text, ocr=used_ocr))

    if not any(_alnum_count(p.text) >= MIN_PAGE_ALNUM for p in pages):
        raise PdfNoTextError(
            "The PDF has no text layer (it looks like a scan). "
            "Turn on OCR (OCR_ENABLED=true; see requirements-ocr.txt) and upload again."
        )
    return pages


# ------------------------------------------------------------------- chunking

# A sentence ends at . ! ? followed by a space and something that starts a
# sentence. Abbreviations ("Fig. 3-1") make this imperfect; chunk boundaries
# only need to be roughly sensible, and overlap covers the misses.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Cut a sentence longer than max_chars at spaces."""
    parts, current = [], ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


# A numbered paragraph heading: "6-181. TITANIUM AND ...", "7-45. INTERNAL ...".
# The number must not continue another token ("AC 43.13-2." has a "." before
# "13") and the heading word must be in capitals, which a reference in running
# text ("see 7-39. Then ...") is not. "SECTION 11. OTHER METALS" also counts.
_HEADING = re.compile(
    r"(?<![\w.\-])\d{1,2}-\d{1,3}\.(?=\s+[A-Z][A-Z(])"
    r"|\bSECTION\s+\d{1,2}\.(?=\s+[A-Z])"
)
# Words that put a paragraph-style number in a caption, not a heading:
# "FIGURE 11-30. TIE-DOWN ...", "TABLE 7-3. ...", and "FI GURE" split by the
# PDF's text layer.
_CAPTION_WORDS = frozenset(
    {"FIGURE", "FIGURES", "TABLE", "TABLES", "GURE", "BLE", "FIG"}
)
# A heading only starts a new chunk once the text before it is at least this
# long, so a section title stays with its first paragraph and a run of short
# headings does not turn into a run of tiny chunks.
MIN_BLOCK_CHARS = 200
# A last chunk that would add fewer new characters than this (beyond the
# overlap it repeats) is folded into the chunk before it instead.
MIN_TAIL_CHARS = 200


def _starts_a_paragraph(text: str, start: int) -> bool:
    """Is the heading-shaped number at `start` really a heading?

    A heading follows the end of a sentence (or a closing bracket), the start
    of the page, or a capitalised section title ("SECTION 11. OTHER METALS AND
    ALLOYS 6-179. ..."). Anything else (a page number in a table of contents,
    "Figures 11-11 through 11-13.", "FIGURE 5-3.") is a reference or caption.
    """
    before = text[:start].rstrip()
    if not before:
        return True
    if before[-1] in ".!?:)]\"'":
        return True
    word = before.split()[-1]
    return word.isalpha() and word.isupper() and word not in _CAPTION_WORDS


def split_paragraphs(text: str, min_block: int = MIN_BLOCK_CHARS) -> list[str]:
    """Cut a page's text at numbered paragraph headings.

    A cut is made only when the text since the previous cut is at least
    `min_block` characters long. Every character of `text` ends up in exactly
    one block (apart from whitespace at the cuts).
    """
    blocks: list[str] = []
    last = 0
    for match in _HEADING.finditer(text):
        start = match.start()
        if start - last < min_block or not _starts_a_paragraph(text, start):
            continue
        blocks.append(text[last:start].strip())
        last = start
    blocks.append(text[last:].strip())
    return [block for block in blocks if block]


def chunk_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
    *,
    paragraphs: bool = False,
) -> list[str]:
    """Cut one page of text into chunks of up to max_chars on sentence ends.

    `paragraphs=True`: first cut the page at numbered paragraph headings, then
    chunk each paragraph on its own (overlap never crosses a heading), and
    fold a last chunk that would add under MIN_TAIL_CHARS of new text into the
    chunk before it, which may then run up to MIN_TAIL_CHARS over max_chars.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not paragraphs:
        return _chunk_sentences(text, max_chars, overlap_chars, merge_tail=0)
    chunks: list[str] = []
    for block in split_paragraphs(text):
        chunks.extend(
            _chunk_sentences(block, max_chars, overlap_chars, merge_tail=MIN_TAIL_CHARS)
        )
    return chunks


def _chunk_sentences(
    text: str, max_chars: int, overlap_chars: int, merge_tail: int
) -> list[str]:
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            sentences.extend(_hard_split(sentence, max_chars))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    new_in_current = 0  # sentences in `current` that no earlier chunk contains
    size = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and size + added > max_chars:
            chunks.append(" ".join(current))
            # carry the tail of this chunk into the next one as overlap
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) + 1 > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current = tail
            new_in_current = 0
            size = sum(len(s) + 1 for s in current) - (1 if current else 0)
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        new_in_current += 1
        size += added
    if current and new_in_current:
        fresh = current[len(current) - new_in_current :]
        if chunks and merge_tail and len(" ".join(fresh)) < merge_tail:
            # A sliver (overlap plus a sentence or two) embeds as a near-copy of
            # the chunk before it and can outrank the chunk holding the answer.
            chunks[-1] = f"{chunks[-1]} {' '.join(fresh)}"
        else:
            chunks.append(" ".join(current))
    return chunks


def chunking_name(paragraphs: bool) -> str:
    """How a document was cut; stored on each chunk so results can be compared."""
    return "paragraphs" if paragraphs else "sentences"


def chunk_pages(
    pages: list[PageText],
    max_chars: int = 1200,
    overlap_chars: int = 150,
    *,
    paragraphs: bool = False,
) -> list[TextChunk]:
    """Chunk every page; a chunk never spans two pages."""
    chunks: list[TextChunk] = []
    for page in pages:
        for text in chunk_text(
            page.text, max_chars, overlap_chars, paragraphs=paragraphs
        ):
            if _alnum_count(text) < MIN_CHUNK_ALNUM:
                continue
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    page=page.number,
                    page_label=page.label,
                    text=text,
                )
            )
    return chunks
