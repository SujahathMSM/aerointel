"""Ingest PDFs from the command line: read, chunk, embed, store.

    python scripts/ingest_pdfs.py data/aviation/AC_43.13-1B.pdf
    python scripts/ingest_pdfs.py data/aviation              # every .pdf in a folder
    python scripts/ingest_pdfs.py manual.pdf --title "AC 43.13-1B"
    python scripts/ingest_pdfs.py data/aviation --dry-run    # read and chunk only
    python scripts/ingest_pdfs.py manual.pdf --replace       # re-chunk a stored file
    python scripts/ingest_pdfs.py manual.pdf --revision "AC 43.13-1B CHG 1"

The revision ("AC 43.13-1B CHG 1") and each page's issue date are read from
the running headers; --dry-run prints what was found, so check it before
storing. --revision / --document-key override the parsed values (one file only).

Same code path as POST /v1/documents (app/services/documents.py:ingest_pdf), but
with no HTTP request to keep open, which is what you want for a 700-page manual.
A file whose bytes were ingested before is skipped, so re-running a folder is
safe. --replace re-ingests such a file with the current chunking settings
(CHUNK_BY_PARAGRAPH, PDF_REPAIR_SPLIT_WORDS, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS): the stored copy and
its chunks are deleted in the same transaction that stores the new ones.

Requires Postgres (migrated: `alembic upgrade head`) and Ollama with the
embedding model pulled. --dry-run needs neither.
"""

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

# Run as `python scripts/ingest_pdfs.py` from the repo root: add the root so
# `app` imports (same approach as backfill_embeddings.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.ocr import ocr_pages  # noqa: E402
from app.services.pdf_text import (  # noqa: E402
    PdfError,
    chunk_pages,
    chunking_name,
    document_key,
    document_revision,
    extract_pages,
    latest_date,
)


def find_pdfs(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.pdf")))
        elif path.is_file():
            found.append(path)
        else:
            print(f"not found: {path}", file=sys.stderr)
    return found


def describe_revision(pages, given: str | None = None, key: str | None = None) -> str:
    """What ingest would store: revision, date, key, and each page's issue."""
    revision, revision_date = document_revision(
        [p.header for p in pages], given=given, fallback=None
    )
    key = key or document_key(revision)
    issues = Counter(str(latest_date(p.header) or "no date") for p in pages)
    found = revision or "(none found: the title will be used)"
    dated = f", dated {revision_date}" if revision_date else ""
    lines = [
        f"  revision: {found}{dated}; document key: {key or '-'}",
        "  page issues: "
        + ", ".join(f"{d}: {n} pages" for d, n in sorted(issues.items())),
    ]
    sample = next((p for p in pages if p.header), None)
    if sample is not None:
        lines.append(f"  header on PDF page {sample.number}: {sample.header!r}")
    return "\n".join(lines)


def dry_run(path: Path, revision: str | None = None, key: str | None = None) -> None:
    settings = get_settings()
    started = time.monotonic()
    pages = extract_pages(
        path.read_bytes(),
        ocr_pages if settings.ocr_enabled else None,
        settings.ocr_language,
        settings.pdf_repair_split_words,
    )
    chunks = chunk_pages(
        pages,
        settings.chunk_max_chars,
        settings.chunk_overlap_chars,
        paragraphs=settings.chunk_by_paragraph,
    )
    labelled = sum(1 for p in pages if p.label)
    ocred = sum(1 for p in pages if p.ocr)
    print(
        f"  {len(pages)} pages ({labelled} with a printed page number, {ocred} OCR), "
        f"{len(chunks)} chunks ({chunking_name(settings.chunk_by_paragraph)}"
        f"{', split words repaired' if settings.pdf_repair_split_words else ''}), "
        f"read in {time.monotonic() - started:.1f}s"
    )
    print(describe_revision(pages, revision, key))
    if chunks:
        sample = chunks[len(chunks) // 2]
        label = f", printed page {sample.page_label}" if sample.page_label else ""
        print(f"  sample (PDF page {sample.page}{label}): {sample.text[:200]!r}")


async def ingest(
    paths: list[Path],
    title: str | None,
    replace: bool = False,
    revision: str | None = None,
    key: str | None = None,
) -> int:
    from app.db.base import SessionFactory, engine
    from app.llm.ollama import OllamaClient
    from app.services.documents import ingest_pdf

    ollama = OllamaClient()
    failures = 0
    try:
        for path in paths:
            print(f"{path}")
            started = time.monotonic()
            last_printed = 0.0

            def progress(done: int, total: int) -> None:
                nonlocal last_printed
                now = time.monotonic()
                if done == total or now - last_printed >= 2:
                    last_printed = now
                    print(f"  embedded {done}/{total} chunks", end="\r", flush=True)

            try:
                async with SessionFactory() as session:
                    result = await ingest_pdf(
                        session,
                        ollama,
                        path.read_bytes(),
                        path.name,
                        title if len(paths) == 1 else None,
                        progress=progress,
                        replace=replace,
                        revision=revision if len(paths) == 1 else None,
                        key=key if len(paths) == 1 else None,
                    )
            except PdfError as exc:
                failures += 1
                print(f"  SKIPPED: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - report and carry on with the next file
                failures += 1
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                continue

            seconds = time.monotonic() - started
            if result.duplicate:
                print(f"  already ingested as document {result.document_id}: skipped")
            else:
                if result.replaced_document_id is not None:
                    print(
                        f"  replaced document {result.replaced_document_id}" + " " * 20
                    )
                print(
                    f"  document {result.document_id}: {result.pages} pages, "
                    f"{result.chunks} chunks, {result.ocr_pages} OCR pages, "
                    f"{seconds:.0f}s" + " " * 20
                )
                print(
                    f"  revision: {result.revision or '-'}"
                    f" ({result.revision_date or 'no date'}), "
                    f"document key: {result.document_key or '-'}"
                )
    finally:
        await ollama.close()
        await engine.dispose()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into AeroIntel.")
    parser.add_argument("paths", nargs="+", help="PDF files or folders containing PDFs")
    parser.add_argument("--title", help="title for a single file (default: file name)")
    parser.add_argument(
        "--dry-run", action="store_true", help="read and chunk only; store nothing"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="re-ingest files that are already stored, with the current chunking "
        "settings (the stored copy and its chunks are deleted)",
    )
    parser.add_argument(
        "--revision",
        help='revision for a single file, e.g. "AC 43.13-1B CHG 1" (default: parsed '
        "from the running headers)",
    )
    parser.add_argument(
        "--document-key",
        help='what stays the same across revisions, e.g. "AC 43.13-1B" (default: '
        "the revision without its change marker)",
    )
    args = parser.parse_args()

    pdfs = find_pdfs(args.paths)
    if not pdfs:
        print("no PDF files found", file=sys.stderr)
        return 1

    if args.dry_run:
        for path in pdfs:
            print(path)
            try:
                dry_run(
                    path,
                    args.revision if len(pdfs) == 1 else None,
                    args.document_key if len(pdfs) == 1 else None,
                )
            except PdfError as exc:
                print(f"  SKIPPED: {exc}")
        return 0

    failures = asyncio.run(
        ingest(pdfs, args.title, args.replace, args.revision, args.document_key)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
