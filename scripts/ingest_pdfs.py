"""Ingest PDFs from the command line: read, chunk, embed, store.

    python scripts/ingest_pdfs.py data/aviation/AC_43.13-1B.pdf
    python scripts/ingest_pdfs.py data/aviation              # every .pdf in a folder
    python scripts/ingest_pdfs.py manual.pdf --title "AC 43.13-1B"
    python scripts/ingest_pdfs.py data/aviation --dry-run    # read and chunk only
    python scripts/ingest_pdfs.py manual.pdf --replace       # re-chunk a stored file

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
    extract_pages,
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


def dry_run(path: Path) -> None:
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
    if chunks:
        sample = chunks[len(chunks) // 2]
        label = f", printed page {sample.page_label}" if sample.page_label else ""
        print(f"  sample (PDF page {sample.page}{label}): {sample.text[:200]!r}")


async def ingest(paths: list[Path], title: str | None, replace: bool = False) -> int:
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
                    print(f"  replaced document {result.replaced_document_id}" + " " * 20)
                print(
                    f"  document {result.document_id}: {result.pages} pages, "
                    f"{result.chunks} chunks, {result.ocr_pages} OCR pages, "
                    f"{seconds:.0f}s" + " " * 20
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
    args = parser.parse_args()

    pdfs = find_pdfs(args.paths)
    if not pdfs:
        print("no PDF files found", file=sys.stderr)
        return 1

    if args.dry_run:
        for path in pdfs:
            print(path)
            try:
                dry_run(path)
            except PdfError as exc:
                print(f"  SKIPPED: {exc}")
        return 0

    return 1 if asyncio.run(ingest(pdfs, args.title, args.replace)) else 0


if __name__ == "__main__":
    sys.exit(main())
