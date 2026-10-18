import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk
from app.llm.ollama import OllamaClient
from app.services.ocr import ocr_pages
from app.services.pdf_text import PdfError, chunk_pages, chunking_name, extract_pages

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
    # With replace=True: the id of the stored copy that this ingest deleted.
    replaced_document_id: int | None = None


def _summary(
    document: Document, duplicate: bool, replaced: int | None = None
) -> IngestedDocument:
    return IngestedDocument(
        document_id=document.id,
        title=document.title,
        filename=document.filename,
        pages=document.page_count,
        chunks=document.chunk_count,
        ocr_pages=document.ocr_pages,
        duplicate=duplicate,
        replaced_document_id=replaced,
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
    replace: bool = False,
) -> IngestedDocument:
    """Turn one PDF into searchable chunks that remember their page.

    Steps: skip the file if these exact bytes were ingested before, read and
    clean the text (OCR for blank pages if enabled), cut it into page-bound
    chunks, embed them in batches, then store the document and all its chunks
    in a single transaction -- a file is either fully ingested or not at all.

    `replace=True` re-ingests a file that is already stored (to apply new
    chunking settings): the old copy is deleted, chunks and all, in the same
    transaction that stores the new one, so a failure part-way leaves the old
    copy in place. Its document id changes.

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
    if existing is not None and not replace:
        logger.info("document already ingested", document_id=existing.id)
        return _summary(existing, duplicate=True)
    replaced = existing.id if existing is not None else None

    pages = await asyncio.to_thread(
        extract_pages,
        data,
        ocr_pages if settings.ocr_enabled else None,
        settings.ocr_language,
        settings.pdf_repair_split_words,
    )
    paragraphs = settings.chunk_by_paragraph
    chunks = chunk_pages(
        pages,
        settings.chunk_max_chars,
        settings.chunk_overlap_chars,
        paragraphs=paragraphs,
    )
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
    if replaced is not None:
        # Chunks go with it (ON DELETE CASCADE); the new copy is stored below in
        # this same transaction, so the hash is never missing from the table.
        await session.execute(delete(Document).where(Document.id == replaced))
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
            metadata_={
                "source": title,
                "filename": filename,
                "chunking": chunking_name(paragraphs),
                "word_repair": settings.pdf_repair_split_words,
            },
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
        chunking=chunking_name(paragraphs),
        replaced_document_id=replaced,
    )
    return _summary(document, duplicate=False, replaced=replaced)
