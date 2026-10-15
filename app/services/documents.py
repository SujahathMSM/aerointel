import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk
from app.llm.ollama import OllamaClient
from app.services.ocr import ocr_pages
from app.services.pdf_text import PdfError, chunk_pages, extract_pages

logger = get_logger(__name__)

MAX_FILENAME_CHARS = 255
MAX_TITLE_CHARS = 300


@dataclass
class IngestedDocument:
    document_id: int
    title: str
    filename: str
    pages: int
    chunks: int
    ocr_pages: int
    # True when this exact file (same bytes) was already ingested: nothing new
    # was embedded or stored, and the numbers describe the existing document.
    duplicate: bool


def _summary(document: Document, duplicate: bool) -> IngestedDocument:
    return IngestedDocument(
        document_id=document.id,
        title=document.title,
        filename=document.filename,
        pages=document.page_count,
        chunks=document.chunk_count,
        ocr_pages=document.ocr_pages,
        duplicate=duplicate,
    )


def clean_filename(filename: str) -> str:
    """Keep the file's own name only (no directories), capped in length."""
    name = PurePath(filename.replace("\\", "/")).name.strip() or "document.pdf"
    return name[:MAX_FILENAME_CHARS]


async def find_document_by_hash(
    session: AsyncSession, content_hash: str
) -> Document | None:
    return (
        await session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
    ).scalar_one_or_none()


async def ingest_pdf(
    session: AsyncSession,
    ollama: OllamaClient,
    data: bytes,
    filename: str,
    title: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IngestedDocument:
    """Turn one PDF into searchable chunks that remember their page.

    Steps: skip the file if these exact bytes were ingested before, read and
    clean the text (OCR for blank pages if enabled), cut it into page-bound
    chunks, embed them in batches, then store the document and all its chunks
    in a single transaction -- a file is either fully ingested or not at all.

    The database connection is released before the slow part (reading,
    embedding): a 700-page manual takes minutes, and a pooled connection held
    that long would starve every other request.

    `progress(done, total)` is called after each embedding batch.
    """
    settings = get_settings()
    filename = clean_filename(filename)
    title = (title or PurePath(filename).stem).strip()[:MAX_TITLE_CHARS] or filename
    content_hash = hashlib.sha256(data).hexdigest()

    existing = await find_document_by_hash(session, content_hash)
    await session.commit()  # release the pooled connection
    if existing is not None:
        logger.info("document already ingested", document_id=existing.id)
        return _summary(existing, duplicate=True)

    pages = await asyncio.to_thread(
        extract_pages,
        data,
        ocr_pages if settings.ocr_enabled else None,
        settings.ocr_language,
    )
    chunks = chunk_pages(pages, settings.chunk_max_chars, settings.chunk_overlap_chars)
    if not chunks:
        raise PdfError("No usable text was found in the PDF.")

    embeddings: list[list[float]] = []
    batch_size = max(1, settings.embed_batch_size)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings.extend(await ollama.embed([chunk.text for chunk in batch]))
        if progress is not None:
            progress(len(embeddings), len(chunks))

    document = Document(
        filename=filename,
        title=title,
        content_hash=content_hash,
        byte_size=len(data),
        page_count=len(pages),
        chunk_count=len(chunks),
        ocr_pages=sum(1 for page in pages if page.ocr),
    )
    session.add(document)
    try:
        await session.flush()
    except IntegrityError:
        # Someone uploaded the same file while this one was being processed.
        await session.rollback()
        existing = await find_document_by_hash(session, content_hash)
        await session.commit()
        if existing is None:
            raise
        return _summary(existing, duplicate=True)

    session.add_all(
        DocumentChunk(
            content=chunk.text,
            metadata_={"source": title, "filename": filename},
            embedding=embedding,
            embedding_model=settings.embedding_model,
            document_id=document.id,
            chunk_index=chunk.index,
            page=chunk.page,
            page_label=chunk.page_label,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    )
    await session.commit()
    logger.info(
        "document ingested",
        document_id=document.id,
        pages=len(pages),
        chunks=len(chunks),
        ocr_pages=document.ocr_pages,
    )
    return _summary(document, duplicate=False)
